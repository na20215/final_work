
def read_label_file_for_classify(label_file_path):
    with open(label_file_path, 'r') as f:
        line = f.readline()
    label = line.strip().split()
    this_label = int(label[0])
    if int(this_label) in [0, 1]:
        class_id = 0
    elif int(this_label) in [2, 3]:
        class_id = 1
    else:
        class_id = 2
    x_center = float(label[1])
    y_center = float(label[2])
    width = float(label[3])
    height = float(label[4])
    return [[class_id, x_center, y_center, width, height]]

def read_label_file_for_classify1(label_file_path):
    with open(label_file_path, 'r') as f:
        line = f.readline()
    label = line.strip().split()
    this_label = int(label[0])
    if int(this_label) in [0, 1]:
        class_id = 0
    elif int(this_label) in [2, 3, 4, 5]:
        class_id = 1
    x_center = float(label[1])
    y_center = float(label[2])
    width = float(label[3])
    height = float(label[4])
    return [[class_id, x_center, y_center, width, height]]

def read_label_file_for_classify2(label_file_path):
    with open(label_file_path, 'r') as f:
        line = f.readline()
    label = line.strip().split()
    this_label = int(label[0])
    if int(this_label) in [2, 3]:
        class_id = 0
    elif int(this_label) in [4, 5]:
        class_id = 1
    x_center = float(label[1])
    y_center = float(label[2])
    width = float(label[3])
    height = float(label[4])
    return [[class_id, x_center, y_center, width, height]]

def read_label_file_for_corner(label_file_path):
    with open(label_file_path, 'r') as f:
        line = f.readline()
    label = line.strip().split()
    this_label = int(label[0])
    class_id = this_label
    x_center = float(label[1])
    y_center = float(label[2])
    width = float(label[3])
    height = float(label[4])
    return [[class_id, x_center, y_center, width, height]]

def read_label_file_for_door(label_file_path):
    with open(label_file_path, 'r') as f:
        line = f.readline()
    label = line.strip().split()
    this_label = int(label[0])
    class_id = this_label - 2
    x_center = float(label[1])
    y_center = float(label[2])
    width = float(label[3])
    height = float(label[4])
    return [[class_id, x_center, y_center, width, height]]

def read_label_file_for_window(label_file_path):
    with open(label_file_path, 'r') as f:
        line = f.readline()
    label = line.strip().split()
    this_label = int(label[0])
    class_id = this_label - 4
    x_center = float(label[1])
    y_center = float(label[2])
    width = float(label[3])
    height = float(label[4])
    return [[class_id, x_center, y_center, width, height]]
