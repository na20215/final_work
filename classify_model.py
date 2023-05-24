import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
import os
import glob
import numpy as np
import cv2
import shutil
import dataset_building
from keras.regularizers import l2
import contextlib
from keras.layers import Dropout

image_dir = 'D:\\dataset\\notgrey\\train\\image\\'
image_dir1 = 'D:\\dataset\\notgrey\\train\\image\\'
image_dir2 = 'D:\\dataset\\notgrey\\train\\image\\door_window\\'
label_dir = 'D:\\dataset\\notgrey\\train\\label\\'
classes = ['corner', 'door', 'window']
classes1 = ['corner', 'door_window']
classes2 = ['door', 'window']
class_weight = {0: 48/106, 1: 48/209, 2: 48/152}
class_weight1 = {0: 48/106, 1: 48/361}
class_weight2 = {0: 48/209, 1: 48/152}
batch_size = 32
img_width = 160
img_height = 160
num_channel = 1
train_ratio = 0.8
val_ratio = 0.2


image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
image_files1 = glob.glob(os.path.join(image_dir1, "*.jpg"))
image_files2 = glob.glob(os.path.join(image_dir2, "*.jpg"))
images = len(image_files)
images1 = len(image_files1)
images2 = len(image_files2)
#print("lenth of images:", images)
print("lenth of images:", images1)
print("lenth of images:", images2)
create_dataset = dataset_building.create_dataset_for_classify(image_dir, label_dir, classes, img_width, img_height, num_channel)
dataset = create_dataset
dataset = dataset.shuffle(images, seed=2025869)

create_dataset1 = dataset_building.create_dataset_for_classify1(image_dir1, label_dir, classes1, img_width, img_height, num_channel)
dataset1 = create_dataset1
dataset1 = dataset1.shuffle(images, seed=2025869)

create_dataset2 = dataset_building.create_dataset_for_classify2(image_dir2, label_dir, classes2, img_width, img_height, num_channel)
dataset2 = create_dataset2
dataset2 = dataset2.shuffle(images, seed=2025869)

train_size = int(images * train_ratio)
val_size = int(images * val_ratio)

train_size1 = int(images1 * train_ratio)
val_size1 = int(images1 * val_ratio)

train_size2 = int(images2 * train_ratio)
val_size2 = int(images2 * val_ratio)

#print(images, train_size, val_size)
print(images1, train_size1, val_size1)
print(images1, train_size2, val_size2)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)

train_dataset1 = dataset1.take(train_size1)
val_dataset1 = dataset1.skip(train_size1).take(val_size1)

train_dataset2 = dataset2.take(train_size2)
val_dataset2 = dataset2.skip(train_size2).take(val_size2)

#model = tf.keras.Sequential([
#   tf.keras.layers.InputLayer(input_shape=(img_width, img_height, num_channel)),
#   tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same',),
#   tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#   tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',),
#   tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#   tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
#   tf.keras.layers.MaxPooling2D((2, 2), strides=2),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
#   tf.keras.layers.Dense(len(classes), activation='softmax', kernel_initializer='glorot_uniform')
#])

#model = tf.keras.Sequential([
#    tf.keras.layers.InputLayer(input_shape=(img_width, img_height, num_channel)),
#    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
#    tf.keras.layers.MaxPooling2D((2, 2)),
#    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
#    tf.keras.layers.MaxPooling2D((2, 2)),
#    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
#    tf.keras.layers.MaxPooling2D((2, 2)),
#    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
#    tf.keras.layers.MaxPooling2D((2, 2)),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform'),
#    Dropout(0.5),
#    tf.keras.layers.Dense(len(classes), activation='softmax', kernel_initializer='glorot_uniform')
#])

model1 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(img_width, img_height, num_channel)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
    Dropout(0.5),
    tf.keras.layers.Dense(len(classes1), activation='softmax', kernel_initializer='glorot_uniform')
])

#model2 = tf.keras.Sequential([
#    tf.keras.layers.InputLayer(input_shape=(img_width, img_height, num_channel)),
#    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
#    tf.keras.layers.MaxPooling2D((2, 2)),
#    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
#    tf.keras.layers.MaxPooling2D((2, 2)),
#    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
#    tf.keras.layers.MaxPooling2D((2, 2)),
#    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
#    tf.keras.layers.MaxPooling2D((2, 2)),
#    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
#    tf.keras.layers.MaxPooling2D((2, 2)),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='glorot_uniform'),
#    Dropout(0.5),
#    tf.keras.layers.Dense(len(classes2), activation='softmax', kernel_initializer='glorot_uniform')
#])

#model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

model1.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#model2.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
#history = model.fit(
#    train_dataset.batch(batch_size),
#    epochs=20,
#    validation_data=val_dataset.batch(batch_size),
#    class_weight=class_weight
#)

history1 = model1.fit(
    train_dataset1.batch(batch_size),
    epochs=100,
    validation_data=val_dataset1.batch(batch_size),
    class_weight=class_weight1
)



#history2 = model2.fit(
#    train_dataset2.batch(batch_size),
#    epochs=50,
#    validation_data=val_dataset2.batch(batch_size),
#    class_weight=class_weight2
#)
#model.save('D:\\models\\classify_model.h5')
model1.save('D:\\models\\classify_model111.h5')
#model2.save('D:\\models\\classify_model222.h5')

#plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label='val_accuracy')
#plt.plot(history.history['val_loss'], label='val_loss')
#plt.plot(history.history['loss'], label='loss')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy/Loss')
#plt.ylim([0.0, 1.0])
#plt.legend(loc='upper left')
#pylab.show()
plt.plot(history1.history['accuracy'], label='accuracy')
plt.plot(history1.history['val_accuracy'], label='val_accuracy')
plt.plot(history1.history['val_loss'], label='val_loss')
plt.plot(history1.history['loss'], label='loss')
plt.title('model1')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.0, 1.0])
plt.legend(loc='upper left')
#plt.plot(history2.history['accuracy'], label='accuracy')
#plt.plot(history2.history['val_accuracy'], label='val_accuracy')
#plt.plot(history2.history['val_loss'], label='val_loss')
#plt.plot(history2.history['loss'], label='loss')
#plt.title('model2')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy/Loss')
#plt.ylim([0.0, 1.0])
#plt.legend(loc='upper left')
pylab.show()
