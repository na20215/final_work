import numpy as np
import os
import glob
from sklearn.metrics import classification_report
from sklearn import svm
def test_function(model,test_dataset,label_dir,test_dir,fileReader,img_width,img_height,num_channel,labels_reader,classes):
    y_pred = model.predict(test_dataset.batch(32)).argmax(axis=1)
    y_true = []
    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        image_file = os.path.join(test_dir, os.path.splitext(os.path.basename(label_file))[0] + '.jpg')
        if not os.path.exists(image_file):
            continue
        img = fileReader.read_image_file(image_file, img_width, img_height, num_channel)

        height, width, _ = img.shape
        labels_data = labels_reader(label_file)
        label = np.zeros(len(classes))
        for label_data in labels_data:
            class_name = classes[label_data[0]]
            class_index = classes.index(class_name)
            y_true.append(class_index)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    report = classification_report(y_true, y_pred, target_names=classes)
    return report

def test_function1(model1, model2, test_dataset,label_dir,test_dir,fileReader,img_width,img_height,num_channel,labels_reader,classes):
    y_pred = model1.predict(test_dataset.batch(32)).argmax(axis=1)
    y_true = []
    for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
        image_file = os.path.join(test_dir, os.path.splitext(os.path.basename(label_file))[0] + '.jpg')
        if not os.path.exists(image_file):
            continue
        img = fileReader.read_image_file(image_file, img_width, img_height, num_channel)

        height, width, _ = img.shape
        labels_data = labels_reader(label_file)
        label = np.zeros(len(classes))
        for label_data in labels_data:
            class_name = classes[label_data[0]]
            class_index = classes.index(class_name)
            y_true.append(class_index)
    y_true = np.array(y_true)
    y_pred_updated = y_pred.copy()
    mask = y_pred == 1
    x2 = test_dataset.batch(32)
    y2 = model2.predict(x2).argmax(axis=1)
    y_pred_updated[mask] += y2[mask]
    y_pred_updated = np.array(y_pred_updated)
    print(y_pred)
    print(y2)
    print(y_true)
    report = classification_report(y_true, y_pred_updated, target_names=classes)
    return report


