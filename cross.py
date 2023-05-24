import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import dataset_building

image_dir = 'D:\\dataset\\notgrey\\train\\image\\'
label_dir = 'D:\\dataset\\notgrey\\train\\label\\'

classes = ['corner', 'door', 'window']
class_weight = {0: 48/106, 1: 48/209, 2: 48/152}
batch_size = 32
img_width = 240
img_height = 240
num_channel = 3
num_classes = 3
num_folds = 5
def create_model():
    model = tf.keras.Sequential([
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
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')
    ])

    return model

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold = 0
accuracy_per_fold = []
loss_per_fold = []
images = 467
indices = np.arange(images)
for train_idx, val_idx in kf.split(np.zeros(indices)):
    print('Training for fold', fold+1, '...')

    train_dataset = dataset_building.create_dataset_for_classify(image_dir, label_dir, classes, img_width, img_height, num_channel, indices=train_idx)
    val_dataset = dataset_building.create_dataset_for_classify(image_dir, label_dir, classes, img_width, img_height, num_channel, indices=val_idx)

    model = create_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_dataset.batch(batch_size),
        epochs=50,
        validation_data=val_dataset.batch(batch_size),
        class_weight=class_weight
    )

    loss, accuracy = model.evaluate(val_dataset.batch(batch_size))
    print('Validation accuracy for fold', fold+1, ':', accuracy)
    print('Validation loss for fold', fold+1, ':', loss)

    accuracy_per_fold.append(accuracy)
    loss_per_fold.append(loss)