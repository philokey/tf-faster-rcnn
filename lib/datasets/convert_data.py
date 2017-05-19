# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import cv2
import six
import os.path as osp
import numpy as np
import tensorflow as tf

_NUM_SHARDS = 5

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""
  def __init__(self, image_ext):
    # Initializes function that decodes RGB JPEG data.
    self._decode_image_data = tf.placeholder(dtype=tf.string)
    if image_ext == 'jpg':
      self._decode_image = tf.image.decode_jpeg(self._decode_image_data, channels=3)
    elif image_ext == 'png':
      self._decode_image = tf.image.decode_png(self._decode_image_data, channels=3)
    else:
      raise ValueError('image_ext can only jpg or png')

  def read_image_dims(self, sess, image_data):
    image = self.decode_image(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_image(self, sess, image_data):
    image = sess.run(self._decode_image,
                     feed_dict={self._decode_image_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _dataset_exists(imdb):
  dataset_dir = osp.join(imdb.data_path, 'records')
  split_name = imdb.name
  for shard_id in range(_NUM_SHARDS):
    output_filename = _get_dataset_filename(
        dataset_dir, split_name, shard_id)
    if not tf.gfile.Exists(output_filename):
      return False
  return True

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = '%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def _image_to_tfexample(image_data, image_format, height, width,
                        num_instances, gt_boxes):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_feature(image_data),
      'image/format': _bytes_feature(image_format),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'label/num_instances': _int64_feature(num_instances),  # N
      'label/gt_boxes': _bytes_feature(gt_boxes), # of shape (N, 5), (x1, y1, x2, y2, label)
  }))

def convert_data(imdb):
  if _dataset_exists(imdb):
    print('Dataset files already exist. Exiting without re-creating them.')
    return
  num_per_shard = int(math.ceil(imdb.num_images / float(_NUM_SHARDS)))
  dataset_dir = osp.join(imdb.data_path, 'records')
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)
  split_name = imdb.name
  gt_roidb = imdb.gt_roidb()
  with tf.Graph().as_default():
    # image_reader = ImageReader(imdb.image_ext)
    with tf.Session('') as sess:
      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, imdb.num_images)
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
              i + 1, imdb.num_images, shard_id))
            sys.stdout.flush()
            # Read the filename:
            # image_data = tf.gfile.FastGFile(imdb.image_path_at(i), 'r').read()
            # height, width = image_reader.read_image_dims(sess, image_data)
            image_data = cv2.imread(imdb.image_path_at(i))
            width, height = image_data.shape[:2]
            gt_boxes = gt_roidb[i]['boxes']
            gt_classes = gt_roidb[i]['gt_classes']
            gt_boxes_with_class = np.empty((gt_boxes.shape[0], 5), dtype=np.float32)
            gt_boxes_with_class[:, 0:4] = gt_boxes
            gt_boxes_with_class[:, 4] = gt_classes
            example = _image_to_tfexample(image_data.tostring(), six.b(imdb.image_ext),
                                height, width, gt_boxes.shape[0],
                                gt_boxes_with_class.tostring())
            tfrecord_writer.write(example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()
