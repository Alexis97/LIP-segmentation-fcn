import os
import shutil
import time
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys
import cv2

import fcn8_vgg
import loss
from utils import *
import LIP_reader  

### parameters setting
N_CLASSES = 20
INPUT_SIZE = (384, 384)
BATCH_SIZE = 4
BATCH_I = 1
SHUFFLE = True
RANDOM_SCALE = False
RANDOM_MIRROR = False
ADAM_OPT_RATE= 0.00001
NUM_STEPS = 1000000
SAVE_PRED_EVERY = 10000
PREDICT_EVERY = 1000
PRINT_PRED_EVERY = 10
DATA_PATH = '/versa/Datasets/LIP/'
TRAINVAL_PATH = DATA_PATH + 'TrainVal_images/'
TRAIN_IMAGE_PATH = TRAINVAL_PATH + 'train_images/'
TRAIN_LABEL_PATH = DATA_PATH + 'TrainVal_parsing_annotations/train_segmentations/'
TRAIN_ID_LIST = TRAINVAL_PATH + 'train_id.txt'
SNAPSHOT_DIR = './checkpoint/FCNNet'
LOG_DIR = './logs/FCNNet'

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
					level=logging.INFO,
					stream=sys.stdout)
from tensorflow.python.framework import ops

def train():
	'''Do training
	'''
	# Create queue coordinator.
	coord = tf.train.Coordinator()
	h, w = INPUT_SIZE
	sess = tf.Session()
	
	# Load reader.
	# print ('Load reader.......................................')
	with tf.name_scope("create_inputs"):
		reader = LIP_reader.ImageReader(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, TRAIN_ID_LIST, 
										INPUT_SIZE, N_CLASSES, RANDOM_SCALE, RANDOM_MIRROR, SHUFFLE, coord)
		image_batch, label_batch = reader.dequeue(BATCH_SIZE)

	# # Set image_batch and label_batch as placeholder
	# # logits = tf.placeholder(tf.float32, shape = [BATCH_SIZE, w, h, N_CLASSES])
	# image_batch = tf.placeholder(tf.float32, shape = [BATCH_SIZE, w, h, 3])
	# label_batch = tf.placeholder(tf.float32, shape = [BATCH_SIZE, w, h, N_CLASSES])
	
	# Build FCN network
	fcn = fcn8_vgg.FCN8VGG()
	# fcn.build(image_batch, train=True, num_classes=N_CLASSES, random_init_fc8=True, debug=True)

	# fcn.build(image_batch, train=True, num_classes=N_CLASSES, random_init_fc8=False, debug=True)
	with tf.name_scope("content_vgg"):
		fcn.build(image_batch, num_classes=N_CLASSES, random_init_fc8=False, debug=True)
	
	print('Finished building Network.')

	# Define loss and optimisation parameters.
	with tf.name_scope('loss'):
		logits = fcn.upscore32
		labels = label_batch
		# print (logits)
		# print (labels)
		loss_ = loss.loss(logits, labels, num_classes=N_CLASSES)
		# total_loss = tf.get_collection('losses')	
		loss_summary = tf.summary.scalar("loss", loss_)

	# Summary
	merged = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

	train_op = tf.train.AdamOptimizer(ADAM_OPT_RATE).minimize(loss_)

	# Saver for storing checkpoints of the model.
	saver = tf.train.Saver(max_to_keep=5)

	# Start queue threads.
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	# Initialize
	logging.info("Start Initializing Variables.")
	init = tf.global_variables_initializer()
	sess.run(init)
	
	# print ('Getting para.................')
	# tvars = tf.trainable_variables()
	# root_dir = '/versa/alexissanchez/LIP-FCN-hair-mask/layer_para/'
	# print (tvars)
	# for i in range(1,6):
	# 	for j in range(1,3):
	# 		conv_name = 'conv{:d}_{:d}'.format(i,j)
			
	# 		if not os.path.exists(root_dir+conv_name):
	# 			os.makedirs(root_dir+conv_name)

	# 		filter_name = conv_name + '/filter:0'
	# 		filter_var = sess.run(filter_name)
	# 		print (filter_name)
	# 		print (filter_var.shape)
	# 		for x in range(3):
	# 			for y in range(3):
	# 				np.savetxt('./layer_para/' + filter_name + '_{:d}_{:d}.txt'.format(x,y), filter_var[x,y,:,:] , fmt = '%.3f')

	# 		bias_name = conv_name + '/biases:0'
	# 		bias_var = sess.run(bias_name)
	# 		print (bias_name)
	# 		print (bias_var.shape)
	# 		np.savetxt('./layer_para/' + bias_name + '.txt', bias_var , fmt = '%.3f')

	# var_name = 'score_fr/biases:0'
	# var = sess.run(var_name)
	# print (var_name + '.................')
	# print (var.shape)
	# np.savetxt('./layer_para/'+var_name+'.txt', var , fmt = '%.3f')

	# Checking demo save path 
	demo_dir = os.getcwd() + '/train_demo/train_2/'
	shutil.rmtree(demo_dir)
	subdir_list = ['image','label','predict']
	for subdir in subdir_list:
		if not os.path.exists(demo_dir+subdir):
			os.makedirs(demo_dir+subdir)

	# Iterate over training steps.
	print ('Start training......')	
	for step in range(NUM_STEPS):
		start_time = time.time()
		loss_value = 0

		# Do training process
		tensors = [merged, loss_, train_op, image_batch, label_batch, fcn.pred_up]
		merged_summary, loss_value, _, origin_image, origin_label, pred_up= sess.run(tensors)
		# merged_summary, loss_value, _, pred_up, bgr = sess.run([merged, loss_, train_op, fcn.pred_up, fcn.bgr])
		summary_writer.add_summary(merged_summary, step)

		if step % PRINT_PRED_EVERY == 0:
			duration = time.time() - start_time
			print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

		if step % PREDICT_EVERY == 0:
			print ('	Doing Demo......')

			origin_image = np.array(origin_image, np.int32)
			origin_label = np.array(origin_label, np.int32)
			# print (origin_image.shape)
			# print (origin_label.shape)

			for im in range(BATCH_SIZE):
				# print (origin_image[im].shape)
				image_color = convert_RGB_TO_BGR(origin_image[im])
				label_color = color_label(origin_label[im])
				pred_result = color_image(pred_up[im])
				cv2.imwrite('{:s}/image/image_{:d}_{:d}.png'.format(demo_dir, step,im), image_color)
				cv2.imwrite('{:s}/label/label_{:d}_{:d}.png'.format(demo_dir, step,im), label_color)
				cv2.imwrite('{:s}/predict/predict_{:d}_{:d}.png'.format(demo_dir, step,im), pred_result)

			duration = time.time() - start_time
			print ('	Done. {:.3f} sec'.format(duration))
		

		if step % SAVE_PRED_EVERY == 0:
			print ('	Saving Model......')

			save_path = SNAPSHOT_DIR + '/model.ckpt'
			saver.save(sess,save_path, global_step = step)

			duration = time.time() - start_time
			print ('	Done. {:.3f} sec'.format(duration))

		
		
	coord.request_stop()
	coord.join(threads)


def test():
	print ('Test...............................')
	# Create queue coordinator.
	coord = tf.train.Coordinator()
	h, w = INPUT_SIZE
	sess = tf.Session()

	

	# Load reader
	with tf.name_scope("create_inputs"):
		reader = LIP_reader.ImageReader(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, TRAIN_ID_LIST, 
										INPUT_SIZE, N_CLASSES, RANDOM_SCALE, RANDOM_MIRROR, SHUFFLE, coord)
		image_batch, label_batch = reader.dequeue(BATCH_SIZE)

	# Build FCN network
	# images = tf.placeholder("float")
	# image_batch = tf.expand_dims(images, 0)

	fcn = fcn8_vgg.FCN8VGG()

	with tf.name_scope("content_vgg"):
		fcn.build(image_batch, num_classes=N_CLASSES, random_init_fc8=False, debug=True)
	
	print('Finished building Network.')
	
	# Summary
	merged = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

	# Generate test input
	# img1 = cv2.imread('./test_data/timg.jpg')
	# img1 = cv2.resize(img1, (h,w))
	
	# print ('image shape')
	# print (img1.shape)
	
	
	# Start queue threads.
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	# Init
	init = tf.global_variables_initializer()
	sess.run(init)

	print('Running the Network')
	
	# Do test process	
	# feed_dict = {images: img1}
	#merged_summary, pred_up, pool5 = sess.run([merged, fcn.pred_up, fcn.pool5], feed_dict = feed_dict)
	merged_summary, pred_up, pool5, bgr = sess.run([merged, fcn.pred_up, fcn.pool5, fcn.bgr])
	summary_writer.add_summary(merged_summary, 0)

	# Write results 
	print ('Write results.................')
	print ('Input bgr')
	print (bgr.shape)


	print ('pred_up')
	print (pred_up.shape)
	pred_result = color_image(pred_up[0])
	cv2.imwrite('./test_demo/pred_result.png', pred_result)

	print ('pool5')
	print (pool5.shape)
	np.savetxt('./test_demo/pool5.txt', pool5[0,0,:,:] , fmt = '%.3f') 

	coord.request_stop()
	coord.join(threads)


def predict(sess, fcn, image_batch, label_batch, batch_size, num_classes, step):
	'''Predict a batch of images during training
	'''
	image = sess.run(image_batch)				# [batch_size, h, w, 3]
	predict_image = sess.run(fcn.pred_up)    	# [batch_size, h, w]
	label = sess.run(label_batch)				# [batch_size, h, w, num_classes]

	# score_fr = sess.run(fcn.score_fr) 			# [batch_size, 12, 12, num_classes]
	# pred = sess.run(fcn.pred) 					# [batch_size, 12, 12]
	# upscore_2 = sess.run(fcn.upscore2)			# [batch_size, 24, 24, num_classes]
	# upscore_4 = sess.run(fcn.upscore4)			# [batch_size, 48, 48, num_classes]
	# upscore_32 = sess.run(fcn.upscore32) 		# [batch_size, h, w, num_classes]		

	conv3_2 = sess.run(fcn.conv3_2) 			# [batch_size, h, w, ?]
	print ('conv3_2')
	print (conv3_2.shape)
	c = conv3_2.shape[3]
	conv3_2 = conv3_2.reshape(batch_size,-1,c)
	np.savetxt('./train_demo/conv3_2.txt', conv3_2[0,:,:] , fmt = '%.3f')

	conv3_3 = sess.run(fcn.conv3_3) 			# [batch_size, h, w, ?]
	print ('conv3_3')
	print (conv3_3.shape)
	c = conv3_3.shape[3]
	conv3_3 = conv3_3.reshape(batch_size,-1,c)
	np.savetxt('./train_demo/conv3_3.txt', conv3_3[0,:,:] , fmt = '%.3f')

	# pred = pred.reshape(batch_size, 12*12, num_classes)	
	# score_fr = score_fr.reshape(batch_size,-1,num_classes)
	# upscore_2 = upscore_2.reshape(batch_size, -1, num_classes)	
	# upscore_4 = upscore_4.reshape(batch_size, -1, num_classes)	
	# upscore_32 = upscore_32.reshape(batch_size, -1, num_classes)


	# print ('predict and label shape................................')
	# print (predict_image.shape)
	# print (label.shape)
	predict_image = np.array(predict_image, np.int32)
	label = np.array(label, np.int32)

	# up_color = decode_labels(up, batch_size, num_classes)
	# label_color = decode_labels(label, batch_size, num_classes)

	for im in range(batch_size):
		image_color = color_image(predict_image[im])
		label_color = color_label(label[im])
		cv2.imwrite('./train_demo/image_{:d}_{:d}.png'.format(step,im), image[im])
		cv2.imwrite('./train_demo/predict_{:d}_{:d}.png'.format(step,im), image_color)
		cv2.imwrite('./train_demo/label_{:d}_{:d}.png'.format(step,im), label_color)

		# np.savetxt('./train_demo/score_fr_{:d}_{:d}.txt'.format(step,im), score_fr[im,:,:] , fmt = '%.3f')
		# np.savetxt('./train_demo/pred_{:d}_{:d}.txt'.format(step,im), pred[im,:,:] , fmt = '%.3f')
		# np.savetxt('./train_demo/upscore_2_{:d}_{:d}.txt'.format(step,im), upscore_2[im,:,:] , fmt = '%.3f')




train()
# test()