from PIL import Image
import numpy as np
import tensorflow as tf
import os
import scipy.misc
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

n_classes = 20
# color map
label_colors = [(0,0,0)
        # 0=Background
        ,(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
        # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
        ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
        # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
        ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
        # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
        ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
        # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe

def convert_RGB_TO_BGR(image):
  return cv2.merge([image[:,:,2], image[:,:,1], image[:,:,0]])

def color_label(label, num_classes=20):
  h, w, n = label.shape
  color_lbl = np.zeros([h,w,3])
  label = np.argmax(label, axis = 2)
  for x in range(h):
    for y in range(w):
      color_lbl[x,y] = label_colors[label[x,y]]
  return color_lbl 

def color_image(image, num_classes=20):
  h, w = image.shape
  color_img = np.zeros([h,w,3])
  
  # for x in range(h):
  #   for y in range(w):
  #     color_img[x,y] = label_colors[image[x,y]]
  # return color_img 

  for cls_id in range(num_classes):
        color_img = np.where(image==cls_id, label_colors[cls_id], color_img)
  return color_img


def save(saver, sess, logdir, step):
    '''Save weights.   
    Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
    '''
    if not os.path.exists(logdir):
        os.makedirs(logdir)   
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
      
    if not os.path.exists(logdir):
      os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
        print("Restored model parameters from {}".format(ckpt_name))
        return True
    else:
        return False  
