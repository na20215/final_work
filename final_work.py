import keras.models
import tensorflow as tf
import glob
import os
import numpy as np
import shutil
import classify_Tool

img_width = 160
img_height = 160
num_channel = 3
image_dir = 'D:\\dataset\\notgrey\\train\\image\\'
corner_image_dir = 'D:\\dataset\\notgrey\\train\\image\\corner\\'
door_image_dir = 'D:\\dataset\\notgrey\\train\\image\\door\\'
window_image_dir = 'D:\\dataset\\notgrey\\train\\image\\window\\'

primary_classes = ['corner', 'door', 'window']
corner_status = ['corner_heat_losing', 'corner_not_heat_losing']
door_status = ['door_heat_losing', 'door_not_heat_losing']
window_status = ['window_heat_losing', 'window_not_heat_losing']

classify_model = keras.models.load_model('D:\\models\\classify_model.h5')
corner_judgment_model = keras.models.load_model('D:\\models\\corner_judgment_model.h5')
door_judgment_model = keras.models.load_model('D:\\models\\door_judgment_model.h5')
window_judgment_model = keras.models.load_model('D:\\models\\window_judgment_model.h5')

print('start to classify objects')
classify_Tool.classify_images(image_dir, primary_classes, classify_model, img_width, img_height, num_channel)

#classify_Tool.classify_images(corner_image_dir, corner_status, corner_judgment_model, img_width, img_height, num_channel)

#classify_Tool.classify_images(door_image_dir, door_status, door_judgment_model, img_width, img_height, num_channel)

#classify_Tool.classify_images(window_image_dir, window_status, window_judgment_model, img_width, img_height, num_channel)






