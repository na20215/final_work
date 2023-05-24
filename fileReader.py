import cv2

def read_image_file1(img_file, img_width, img_height, num_channel):# this is for classify
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    return img

def read_image_file(img_file, img_width, img_height, num_channel): # this is for other model
    img = cv2.imread(img_file)
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0
    return img