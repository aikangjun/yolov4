# annotation/get_annotation_info
train_annotation_path = r'D:\dataset\image\COCO\annotations\\' \
                        r'instances_train2017.json'
validation_annotation_path = r'D:\dataset\image\COCO\annotations\\' \
                             r'instances_val2017.json'
category_names_to_detect = ['person', 'boat', 'truck']  # 选出的每类的样本个数应该接近
data_dir = r'C:\Users\chen\Desktop\zvan\yolov4\data_info'
image_dir = r'D:\dataset\image\COCO\image\train2017'
# annotation/kmeans_for_anchors
input_size = (416, 416)  # (h,w)

# _utils
from _utils.utils import get_anchors

max_boxes = 100
batch_size = 4
train_ratio = 0.8
anchors = get_anchors(anchors_path=r'C:\Users\chen\Desktop\zvan\yolov4\data_info\anchors.txt')
classes_name = category_names_to_detect
num_classes = classes_name.__len__()
# network
num_anchors = 3

# draw
font_path = '.\\font\\simhei.ttf'
sample_path = '.\\result\\Batch{}.jpg'

# train
cosine_scheduler = True
Epoches = 30
learning_rate = 1e-5
warmup_learning_rate = 1e-6
min_learning_rate = 1e-7
ckpt_path = '.\\ckpt'
# ckpt_path = '.\\ckpt_2'
