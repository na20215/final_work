import tensorflow as tf
import matplotlib.pyplot as plt
import pylab
import os
import glob
import numpy as np
import cv2
import shutil
import dataset_building
from keras.layers import Dropout
image_dir = 'D:\\dataset\\notgrey\\train\\image\\window\\'
label_dir = 'D:\\dataset\\notgrey\\train\\label\\'
classes = ['window_heat_losing', 'window_not_heat_losing']
batch_size = 32
img_width = 160
img_height = 160
num_channel = 3
train_ratio = 0.8
val_ratio = 0.2
class_weight = {0: 6/24, 1: 18/24}
images_file = glob.glob(os.path.join(image_dir, '*.jpg'))
images = len(images_file)
dataset = dataset_building.create_dataset_for_window(image_dir, label_dir, classes, img_width, img_height, num_channel)
dataset = dataset.shuffle(images, reshuffle_each_iteration=True)

training_dataset_size = int(images * train_ratio)
val_dataset_size = int(images * val_ratio)

training_dataset = dataset.take(training_dataset_size)
val_dataset = dataset.skip(training_dataset_size).take(val_dataset_size)


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(img_width, img_height, num_channel)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',  kernel_initializer='glorot_uniform'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='glorot_uniform'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform'),
    tf.keras.layers.Dense(2, activation='softmax', kernel_initializer='glorot_uniform')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(
    training_dataset.batch(batch_size),
    epochs=50,
    validation_data=val_dataset.batch(batch_size),
    class_weight=class_weight
)

model.save('D:\\models\\window_judgment_model.h5')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.ylim([0.0, 1.0])
plt.legend(loc='upper left')
pylab.show()