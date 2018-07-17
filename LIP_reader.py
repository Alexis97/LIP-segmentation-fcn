import os

import numpy as np
import tensorflow as tf

IGNORE_LABEL = 255
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def image_scaling(img, label):
	"""
	Randomly scales the images between 0.5 to 1.5 times the original size.

	Args:
	  img: Training image to scale.
	  label: Segmentation mask to scale.
	"""
	
	scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
	h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
	w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
	new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
	img = tf.image.resize_images(img, new_shape)
	label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
	label = tf.squeeze(label, squeeze_dims=[0])
   
	return img, label

def image_mirroring(img, label):
	"""
	Randomly mirrors the images.

	Args:
	  img: Training image to mirror.
	  label: Segmentation mask to mirror.
	"""
	# print ('image mirror.............')
	# print (img)
	distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
	mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
	img = tf.reverse(img, mirror)
	reversed_label = tf.reverse(label, mirror)

	return img, reversed_label

def random_resize_img_labels(image, label, resized_h, resized_w):

	scale = tf.random_uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
	h_new = tf.to_int32(tf.multiply(tf.to_float(resized_h), scale))
	w_new = tf.to_int32(tf.multiply(tf.to_float(resized_w), scale))

	new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
	img = tf.image.resize_images(image, new_shape)
	label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
	label = tf.squeeze(label, squeeze_dims=[0])
	return img, label

def resize_img_labels(image, label, resized_h, resized_w):

	new_shape = tf.stack([tf.to_int32(resized_h), tf.to_int32(resized_w)])
	img = tf.image.resize_images(image, new_shape)
	label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
	label = tf.squeeze(label, squeeze_dims=[0])
	return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
	"""
	Randomly crop and pads the input images.

	Args:
	  image: Training image to crop/ pad.
	  label: Segmentation mask to crop/ pad.
	  crop_h: Height of cropped segment.
	  crop_w: Width of cropped segment.
	  ignore_label: Label to ignore during the training.
	"""

	label = tf.cast(label, dtype=tf.float32)
	label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
	combined = tf.concat([image, label], 2) 
	image_shape = tf.shape(image)
	combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
	
	last_image_dim = tf.shape(image)[-1]
	last_label_dim = tf.shape(label)[-1]
	combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
	img_crop = combined_crop[:, :, :last_image_dim]
	label_crop = combined_crop[:, :, last_image_dim:]
	label_crop = label_crop + ignore_label
	label_crop = tf.cast(label_crop, dtype=tf.uint8)
	
	# Set static shape so that tensorflow knows shape at compile time. 
	img_crop.set_shape((crop_h, crop_w, 3))
	label_crop.set_shape((crop_h,crop_w, 1))
	return img_crop, label_crop  



def convert_id_to_path(image_list_path, image_dir, label_dir):
	"""Convert image id to image path and label path.
	
	Args:
	  image_list_path: path to image id list, eg. '1000_1234574', '100111_470108'.
	  image_dir: path to the directory with images.
	  label_dir: path to the directory with labels.
	   
	Returns:
	  Two lists with all file names for images and labels, respectively.
	"""

	f = open(image_list_path, 'r')
	images = []
	labels = []
	for line in f:
		image_id = line.replace('\n','')
		# print ('Image ID.........................')
		# print (image_id)
		images.append(image_dir + image_id + '.jpg')
		labels.append(label_dir + image_id + '.png')
	f.close()
	return images, labels

def read_images_from_disk(input_queue, input_size, num_classes, random_scale, random_mirror): # optional pre-processing arguments
	"""Read one image and its corresponding label with optional pre-processing.
	
	Args:
	  input_queue: tf queue with paths to the image and its label.
	  input_size: a tuple with (height, width) values.
				  If not given, return images of original size.
	  num_classes: number of classes
	  random_scale: whether to randomly scale the images prior
					to random crop.
	  random_mirror: whether to randomly mirror the images prior
					to random crop.
	  
	Returns:
	  Two tensors: the decoded image and its label.
	"""

	image_contents = tf.read_file(input_queue[0])
	label_contents = tf.read_file(input_queue[1])

	img = tf.image.decode_jpeg(image_contents, channels=3)
	# img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
	# img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
	# # Extract mean.
	# img -= IMG_MEAN

	label = tf.image.decode_png(label_contents, channels=1)
	

	if input_size is not None:
		h, w = input_size

		# Randomly mirror the images and labels.
		if random_mirror:
			img, label = image_mirroring(img, label)

		# Randomly resize the images and labels.
		if random_scale:
			img, label = random_resize_img_labels(img, label)
			# Random scale must be followed by crop to create fixed size
			img, label = random_crop_and_pad_image_and_labels(img, label, h, w, IGNORE_LABEL)
		else:
			img, label = resize_img_labels(img, label, h, w)

	# convert label to one hot map
	# print ('Label 1')
	# print (label)
	label = tf.one_hot(label, depth = num_classes)
	label = tf.reshape(label, [h, w, num_classes])
	# print ('Label 2')
	# print (label)
	return img, label

class ImageReader(object):
	'''Generic ImageReader which reads images and corresponding segmentation
	   masks from the disk, and enqueues them into a TensorFlow queue.
	'''

	def __init__(self, image_dir, label_dir, image_list_path, input_size, num_classes, random_scale,
				 random_mirror, shuffle, coord):
		'''Initialise an ImageReader.
		
		Args:
		  image_dir: path to the directory with images
		  label_dir: path to the directory with labels
		  image_list_path: path to a list contains image ids
		  input_size: a tuple with (height, width) values, to which all the images will be resized.
		  num_classes: number of classes
		  random_scale: whether to randomly scale the images prior to random crop.
		  random_mirror: whether to randomly mirror the images prior to random crop.
		  coord: TensorFlow queue coordinator.
		'''
		self.image_dir = image_dir
		self.label_dir = label_dir
		self.image_list_path = image_list_path
		self.input_size = input_size
		self.num_classes = num_classes
		self.coord = coord

		self.image_list, self.label_list = convert_id_to_path(self.image_list_path, self.image_dir, self.label_dir)
		self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
		self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
		self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=shuffle) 
		self.image, self.label = read_images_from_disk(self.queue, self.input_size, self.num_classes, random_scale, random_mirror) 

	def dequeue(self, num_elements):
		'''Pack images and labels into a batch.
		
		Args:
		  num_elements: the batch size.
		  
		Returns:
		  Two tensors of size (batch_size, h, w, {3, num_classes}) for images and masks.'''
		image_batch, label_batch = tf.train.batch([self.image, self.label],
												  num_elements)
		return image_batch, label_batch
