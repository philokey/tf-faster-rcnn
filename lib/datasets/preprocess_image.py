from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf
from model.config import cfg

from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import slim


def _crop(image, offset_height, offset_width, crop_height, crop_width):
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  cropped_shape = control_flow_ops.with_dependencies(
      [rank_assertion],
      tf.stack([crop_height, crop_width, original_shape[2]]))

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  image = control_flow_ops.with_dependencies(
      [size_assertion],
      tf.slice(image, offsets, cropped_shape))
  return tf.reshape(image, cropped_shape)


def _random_crop(image_list, label_list, crop_height, crop_width):
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  image_shape = control_flow_ops.with_dependencies(
      [rank_assertions[0]],
      tf.shape(image_list[0]))
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.', image_height, image_width, crop_height, crop_width])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                               tf.shape(image))
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  max_offset_height = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_height - crop_height + 1, []))
  max_offset_width = control_flow_ops.with_dependencies(
      asserts, tf.reshape(image_width - crop_width + 1, []))
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  cropped_images = [_crop(image, offset_height, offset_width,
                          crop_height, crop_width) for image in image_list]
  cropped_labels = [_crop(label, offset_height, offset_width,
                          crop_height, crop_width) for label in label_list]
  return cropped_images, cropped_labels


def _central_crop(image_list, label_list, crop_height, crop_width):
  output_images = []
  output_labels = []
  for image, label in zip(image_list, label_list):
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    output_images.append(_crop(image, offset_height, offset_width,
                               crop_height, crop_width))
    output_labels.append(_crop(label, offset_height, offset_width,
                               crop_height, crop_width))
  return output_images, output_labels


def _smallest_size_at_least(height, width, smallest_side):
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width

def _aspect_preserving_resize(image, label, smallest_side):
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)

  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image, axis=[0])
  resized_image.set_shape([None, None, 3])

  label = tf.expand_dims(label, 0)
  resized_label = tf.image.resize_nearest_neighbor(label, [new_height, new_width],
                                                   align_corners=False)
  resized_label = tf.squeeze(resized_label, axis=[0])
  resized_label.set_shape([None, None, 1])
  return resized_image, resized_label

def flip_gt_boxes(gt_boxes, ih, iw):
    x1s, y1s, x2s, y2s, cls = \
            gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3], gt_boxes[:, 4]
    x1s = tf.to_float(iw) - x1s
    x2s = tf.to_float(iw) - x2s
    return tf.concat(values=(x2s[:, tf.newaxis],
                             y1s[:, tf.newaxis],
                             x1s[:, tf.newaxis],
                             y2s[:, tf.newaxis],
                             cls[:, tf.newaxis]), axis=1)

def flip_gt_masks(gt_masks):
    return tf.reverse(gt_masks, axis=[2])

def flip_image(image):
    return tf.reverse(image, axis=[1])

def resize_gt_boxes(gt_boxes, scale_ratio):
    xys, cls = \
            gt_boxes[:, 0:4], gt_boxes[:, 4]
    xys = xys * scale_ratio
    return tf.concat(values=(xys, cls[:, tf.newaxis]), axis=1)


def preprocess_image(image, gt_boxes, is_training=False):
  """preprocess image for coco
  1. random flipping
  2. min size resizing
  3. zero mean
  4. ...
  """
  if is_training:
    return preprocess_for_training(image, gt_boxes)
  else:
    return preprocess_for_test(image, gt_boxes)


def preprocess_for_training(image, gt_boxes):
  ih, iw = tf.shape(image)[0], tf.shape(image)[1]
  ## random flipping
  coin = tf.to_float(tf.random_uniform([1]))[0]
  image, gt_boxes = \
    tf.cond(tf.greater_equal(coin, 0.5),
            lambda: (flip_image(image),
                     flip_gt_boxes(gt_boxes, ih, iw)),
            lambda: (image, gt_boxes))

  ## min size resizing
  random_scale_inds = random.randint(0, len(cfg.TRAIN.SCALES) - 1)
  image_min_size = cfg.TRAIN.SCALES[random_scale_inds]
  new_ih, new_iw =_smallest_size_at_least(ih, iw, image_min_size)
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [new_ih, new_iw], align_corners=False)
  image = tf.squeeze(image, axis=[0])

  scale_ratio = tf.to_float(new_ih) / tf.to_float(ih)
  gt_boxes = resize_gt_boxes(gt_boxes, scale_ratio)

  ## rgb to bgr
  image = tf.reverse(image, axis=[-1])

  ## zero mean image
  image = tf.cast(image, tf.float32)
  # image = image - cfg.PIXEL_MEANS
  mean = tf.constant(cfg.PIXEL_MEANS, dtype=tf.float32)
  mean = tf.reshape(mean, [1, 1, 3])
  image = image - mean

  image = tf.expand_dims(image, axis=0)

  return image, gt_boxes


def preprocess_for_test(image, gt_boxes):
  ih, iw = tf.shape(image)[0], tf.shape(image)[1]

  ## min size resizing
  new_ih, new_iw = _smallest_size_at_least(ih, iw, cfg.TEST.SCALES[0])
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_bilinear(image, [new_ih, new_iw], align_corners=False)
  image = tf.squeeze(image, axis=[0])

  scale_ratio = tf.to_float(new_ih) / tf.to_float(ih)
  gt_boxes = resize_gt_boxes(gt_boxes, scale_ratio)

  ## rgb to bgr
  image = tf.reverse(image, axis=[-1])

  ## zero mean image
  image = tf.cast(image, tf.float32)
  # image = image / 256.0
  # image = (image - 0.5) * 2.0
  image = tf.expand_dims(image, axis=0)
  mean = tf.constant(cfg.PIXEL_MEANS, dtype=tf.float32)
  mean = tf.reshape(mean, [1, 1, 3])
  return image, gt_boxes

