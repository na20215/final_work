import glob
import os
import fileReader
import labels_reader
import numpy as np
import tensorflow as tf

def create_dataset_for_classify(image_dir, label_dir, classes, img_width, img_height, num_channel):
    images = []
    labels = []
    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        image_file = os.path.join(image_dir, os.path.splitext(os.path.basename(label_file))[0] + '.jpg')
        if not os.path.exists(image_file):
            continue
        img = fileReader.read_image_file1(image_file, img_width, img_height, num_channel)
        height, width = img.shape
        labels_data = labels_reader.read_label_file_for_classify(label_file)
        label = np.zeros(len(classes))
        for label_data in labels_data:
            class_name = classes[label_data[0]]
            class_index = classes.index(class_name)
            label[class_index] = 1
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

def create_dataset_for_classify1(image_dir, label_dir, classes, img_width, img_height, num_channel):
    images = []
    labels = []
    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        image_file = os.path.join(image_dir, os.path.splitext(os.path.basename(label_file))[0] + '.jpg')
        if not os.path.exists(image_file):
            continue
        img = fileReader.read_image_file1(image_file, img_width, img_height, num_channel)
        height, width = img.shape
        labels_data = labels_reader.read_label_file_for_classify1(label_file)
        label = np.zeros(len(classes))
        for label_data in labels_data:
            class_name = classes[label_data[0]]
            class_index = classes.index(class_name)
            label[class_index] = 1
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

def create_dataset_for_classify2(image_dir, label_dir, classes, img_width, img_height, num_channel):
    images = []
    labels = []
    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        image_file = os.path.join(image_dir, os.path.splitext(os.path.basename(label_file))[0] + '.jpg')
        if not os.path.exists(image_file):
            continue
        img = fileReader.read_image_file1(image_file, img_width, img_height, num_channel)
        height, width = img.shape
        labels_data = labels_reader.read_label_file_for_classify2(label_file)
        label = np.zeros(len(classes))
        for label_data in labels_data:
            class_name = classes[label_data[0]]
            class_index = classes.index(class_name)
            label[class_index] = 1
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

def create_dataset_for_corner(image_dir, label_dir, classes, img_width, img_height, num_channel):
    images = []
    labels = []
    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        image_file = os.path.join(image_dir, os.path.splitext(os.path.basename(label_file))[0] + '.jpg')
        if not os.path.exists(image_file):
            continue
        img = fileReader.read_image_file(image_file, img_width, img_height, num_channel)

        height, width, _ = img.shape
        labels_data = labels_reader.read_label_file_for_corner(label_file)
        label = np.zeros(len(classes))
        for label_data in labels_data:
            class_name = classes[label_data[0]]

            class_index = classes.index(class_name)
            label[class_index] = 1
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

def create_dataset_for_door(image_dir, label_dir, classes, img_width, img_height, num_channel):
    images = []
    labels = []
    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        image_file = os.path.join(image_dir, os.path.splitext(os.path.basename(label_file))[0] + '.jpg')
        if not os.path.exists(image_file):
            continue
        img = fileReader.read_image_file(image_file, img_width, img_height, num_channel)

        height, width, _ = img.shape
        labels_data = labels_reader.read_label_file_for_door(label_file)
        label = np.zeros(len(classes))
        for label_data in labels_data:
            class_name = classes[label_data[0]]
            class_index = classes.index(class_name)
            label[class_index] = 1
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset


def create_dataset_for_window(image_dir, label_dir, classes, img_width, img_height, num_channel):
    images = []
    labels = []
    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        image_file = os.path.join(image_dir, os.path.splitext(os.path.basename(label_file))[0] + '.jpg')
        if not os.path.exists(image_file):
            continue
        img = fileReader.read_image_file(image_file, img_width, img_height, num_channel)

        height, width, _ = img.shape
        labels_data = labels_reader.read_label_file_for_window(label_file)
        label = np.zeros(len(classes))
        for label_data in labels_data:
            class_name = classes[label_data[0]]
            class_index = classes.index(class_name)
            label[class_index] = 1
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

