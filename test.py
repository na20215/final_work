import dataset_building
import keras
import numpy as np
import labels_reader
import glob
import os
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import fileReader
import test_function
batch_size = 32
img_width = 160
img_height = 160
num_channel = 3
label_dir = 'D:\\dataset\\notgrey\\train\\label\\'

primary_classes = ['corner', 'door', 'window']
primary_classes1 = ['corner', 'door_window']
primary_classes2 = ['door', 'window']
corner_status = ['corner_heat_losing', 'corner_not_heat_losing']
door_status = ['door_heat_losing', 'door_not_heat_losing']
window_status = ['window_heat_losing', 'window_not_heat_losing']

classify_test_dir = 'D:\\dataset\\test'
classify_test_dir1 = 'D:\\dataset\\test'
classify_test_dir2 = 'D:\\dataset\\test\\door_window'
corner_test_dir = 'D:\\dataset\\test\\corner'
door_test_dir = 'D:\\dataset\\test\\door'
window_test_dir = 'D:\\dataset\\test\\window'
image_dir = 'D:\\dataset\\notgrey\\train\\image\\'
corner_image_dir = 'D:\\dataset\\notgrey\\train\\image\\corner\\'
door_image_dir = 'D:\\dataset\\notgrey\\train\\image\\door\\'
window_image_dir = 'D:\\dataset\\notgrey\\train\\image\\window\\'

classify_model = keras.models.load_model('D:\\models\\classify_model.h5')
classify_model1 = keras.models.load_model('D:\\models\\classify_model1111.h5')
classify_model2 = keras.models.load_model('D:\\models\\classify_model2222.h5')
corner_judgment_model = keras.models.load_model('D:\\models\\corner_judgment_model.h5')
door_judgment_model = keras.models.load_model('D:\\models\\door_judgment_model.h5')
window_judgment_model = keras.models.load_model('D:\\models\\window_judgment_model.h5')

create_test_dataset = dataset_building.create_dataset_for_classify(classify_test_dir, label_dir, primary_classes, img_width, img_height, num_channel)
create_test_dataset1 = dataset_building.create_dataset_for_classify1(classify_test_dir1, label_dir, primary_classes1, img_width, img_height, num_channel)
create_test_dataset2 = dataset_building.create_dataset_for_classify2(classify_test_dir2, label_dir, primary_classes2, img_width, img_height, num_channel)
create_corner_test_dataset = dataset_building.create_dataset_for_corner(corner_test_dir, label_dir, corner_status, img_width, img_height, num_channel)
create_door_test_dataset = dataset_building.create_dataset_for_door(door_test_dir, label_dir, door_status, img_width, img_height, num_channel)
create_window_test_dataset = dataset_building.create_dataset_for_window(window_test_dir, label_dir, window_status, img_width, img_height, num_channel)
test_dataset = create_test_dataset
test_dataset1 = create_test_dataset1
test_dataset2 = create_test_dataset2
corner_test_dataset = create_corner_test_dataset
door_test_dataset = create_door_test_dataset
window_test_dataset = create_window_test_dataset



#classify_report = test_function.test_function(classify_model, test_dataset,
#                                             label_dir, classify_test_dir, fileReader, img_width,
#                                             img_height, num_channel,
#                                             labels_reader.read_label_file_for_classify,
#                                             primary_classes)



#classify_report1 = test_function.test_function(classify_model1, test_dataset1,
#                                              label_dir, classify_test_dir1, fileReader, img_width,
#                                              img_height, num_channel,
#                                              labels_reader.read_label_file_for_classify1,
#                                              primary_classes1)
#
#classify_report2 = test_function.test_function(classify_model2, test_dataset2,
#                                              label_dir, classify_test_dir2, fileReader, img_width,
#                                              img_height, num_channel,
#                                              labels_reader.read_label_file_for_classify2,
#                                              primary_classes2)

#classify_report3 = test_function.test_function1(classify_model1, classify_model2, test_dataset1,
#                                                label_dir, classify_test_dir1, fileReader, img_width,
#                                                img_height, num_channel,
#                                                labels_reader.read_label_file_for_classify,
#                                                primary_classes)

corner_jugment_report = test_function.test_function(corner_judgment_model, corner_test_dataset,
                                                    label_dir, corner_test_dir, fileReader, img_width,
                                                    img_height, num_channel,
                                                   labels_reader.read_label_file_for_corner,
                                                    corner_status)

#door_jugment_report = test_function.test_function(door_judgment_model, door_test_dataset,
#                                                    label_dir, door_test_dir, fileReader, img_width,
#                                                    img_height, num_channel,
#                                                    labels_reader.read_label_file_for_door,
#                                                    door_status)

#window_jugment_report = test_function.test_function(door_judgment_model, window_test_dataset,
#                                                    label_dir, window_test_dir, fileReader, img_width,
#                                                    img_height, num_channel,
#                                                    labels_reader.read_label_file_for_window,
#                                                    window_status)

#print('classify_report')
#print(classify_report)
#print('classify_report1')
#print(classify_report1)
#print('classify_report2')
#print(classify_report2)
#print('classify_report3')
#print(classify_report3)
print('corner_report')
print(corner_jugment_report)
#print('door_jugment_report')
#print(door_jugment_report)
#print('window_jugment_report')
#print(window_jugment_report)