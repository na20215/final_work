classes = ['corner_heat_losing', 'corner_not_heat_losing', 'doors_heat_losing','doors_not_heat_losing','windows_heat_losing','windows_not_heat_losing']
import os
import fileReader
import numpy as np
import shutil

def classify_images(image_dir, classes, model, img_width, img_height, num_channel):
    for c in classes:
        path = os.path.join(image_dir, c)
        os.makedirs(path, exist_ok=True)

    for image_file in os.listdir(image_dir):
        if os.path.isdir(os.path.join(image_dir, image_file)):
            continue
        image_path = os.path.join(image_dir, image_file)
        img = fileReader.read_image_file1(image_path, img_width, img_height, num_channel)
        prediction = model.predict(np.array([img]))
        class_id = np.argmax(prediction)
        class_name = classes[class_id]
        src_path = os.path.join(image_dir, image_file)
        dst_path = os.path.join(image_dir, class_name, image_file)
        shutil.move(src_path, dst_path)



