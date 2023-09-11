import numpy as np
import pandas as pd
import json
import configure.config as cfg
import tensorflow.keras as keras
import os


def get_annotation_dict_coco(dataset_type: str,
                             category_names_to_detect: list):
    """

    :param dataset_type: str类型，可以为'train'训练集的annotation，或者'validation'验证集的annotation
    :param bbox_area_ascending: bool类型,True或者Fals，bounding_box的面积是否进行从小到达排序，
    :return:
    annotations_dict:dict 处理后得annotation 包括center_point_x,center_point_y,width,
    height,category_id

    annotations_raw_dict:dict json文件的原始数据格式
    """

    # 根据数据集类型拿到训练集注释文件地址或验证集文件地址
    # 使用open()打开文件，并使用json.load()将json转为dict
    # 得到标注文件的原始字典
    if dataset_type == 'train':
        annotations_path = cfg.train_annotation_path
    elif dataset_type == 'validation':
        annotations_path = cfg.validation_annotation_path
    else:
        annotations_path = None
    try:
        with open(annotations_path, mode='r') as f:
            annotations_raw_dict = json.load(f)
    except FileNotFoundError:
        print(f'File not found: {annotations_path}')
    # 定义一个空字典，保存所有目标检查的所有标准信息
    annotations_dict = {}
    # 通过keras.utils.Progbar()定义一个进度条
    progress_bar = keras.utils.Progbar(len(annotations_raw_dict['annotations']),
                                       width=50,
                                       verbose=1,
                                       interval=0.5,
                                       stateful_metrics=None,
                                       unit_name='step')
    print(f'Extracting the annotations for {dataset_type} dataset..')
    # revise_records 保存
    revise_records = []
    for i, each_annotation in enumerate(annotations_raw_dict['annotations']):
        # each_annotation 为dict,包含关键字'segmentation','area','iscrowd','image_id','bbox','category_id','id'
        # 'segmentation'对应value为list,分割掩码像素列表
        # 'area'为float类型,分割掩码内的像素数
        # 'iscrowd':int  针对单个目标(0)还是针对彼此靠近的对各目标(1)
        # 'image_id':int 与'image'字典的'id'字段相同,该值用于'image'字典和其他字典的交叉引用
        # 'bbox':list 边界框 [左上x,左上y,box的width,box的height]
        # 'category_id':int 对应'categories'字典的'id'字段
        # 'id': 为'annotation'字典的唯一标识

        # 使用enumerate得到i,对progress_bar进行update
        progress_bar.update(i)
        # image_id 为int
        image_id = each_annotation['image_id']
        # category_id 为 int
        category_id = each_annotation['category_id']
        # bbox 为 list ,长度为4,分别为左上角点坐标的x，y和bbox的w,h
        bbox = each_annotation['bbox']
        top_left_x = int(bbox[0])
        top_left_y = int(bbox[1])
        box_width = int(bbox[2])
        box_height = int(bbox[3])
        bottom_right_x = top_left_x + box_width
        bottom_right_y = top_left_y + box_height

        if str(format(image_id, '012d')) not in annotations_dict:
            # 将image_id与bbox信息对应起来,放入字典annotation_dict中
            # 如果没有对用的image_id的key,在字典中新建一个空list,进行添加
            first_annotation = [top_left_x, top_left_y,
                                bottom_right_x, bottom_right_y, category_id]
            for each in annotations_raw_dict['categories']:
                for j, name in enumerate(category_names_to_detect):
                    if category_id == each['id'] and name == each['name']:
                        first_annotation[4] = j
                        annotations_dict[str(format(image_id, '012d'))] = []
                        annotations_dict[str(format(image_id, '012d'))].append(
                            first_annotation)
        else:
            # 在dict中有对应的key,直接进行append
            later_annotation = [top_left_x, top_left_y,
                                bottom_right_x, bottom_right_y, category_id]
            for each in annotations_raw_dict['categories']:
                for j, name in enumerate(category_names_to_detect):
                    if category_id == each['id'] and name == each['name']:
                        later_annotation[4] = j
                        annotations_dict[str(format(image_id, '012d'))].append(
                            later_annotation)
    return annotations_dict, annotations_raw_dict


def coco_categories_to_detect(anntations_raw_dict: dict):
    """
    操作categories的values  根据检测得类别名字，将类别名字、类别id、supercategory 3个关联，存为DataFrame
    :return:
    catecategories_to_detect:DataFrame 包含需要检测类的信息，id_in_model,id_in_coco,
    supercategory,name
    full_categories:DataFrame 包含全部80个类别，
    """
    # 创建full_categories
    full_categories = pd.DataFrame({})
    for i, each in enumerate(anntations_raw_dict['categories']):
        id_in_model = i
        full_categories.loc[i, 'id_in_model'] = id_in_model
        full_categories.loc[i, 'id_in_coco'] = each['id']
        full_categories.loc[i, 'supercategory'] = each['supercategory']
        full_categories.loc[i, 'name'] = each['name']
    # full_categor
    categories_to_detect = full_categories[
        full_categories['name'].isin(cfg.category_names_to_detect)]
    # 使用现有列设置DataFrame的索引
    categories_to_detect = categories_to_detect.set_index('id_in_model')
    return categories_to_detect, full_categories


if __name__ == '__main__':
    annotation_dict, annotation_raw_dict = get_annotation_dict_coco(
        dataset_type='train',
        category_names_to_detect=cfg.category_names_to_detect)
    categroies_to_detect, full_categroies = coco_categories_to_detect(
        annotation_raw_dict)
    if not os.path.exists(cfg.data_dir):
        os.mkdir(cfg.data_dir)
    categroies_to_detect.to_csv(
        path_or_buf=cfg.data_dir + '\\categroies_to_detect.csv')
    full_categroies.to_csv(path_or_buf=cfg.data_dir + '\\full_categroies.csv')
    with open(file=cfg.data_dir + '\\annotation.txt', mode='w',
              encoding='utf-8') as f:
        for key, value in annotation_dict.items():
            f.write(str(cfg.image_dir) + '\\' + str(key) + '.jpg' + ' ')
            for v in value:
                for i, x in enumerate(v):
                    if i == v.__len__() - 1:
                        f.write(str(x) + ' ')
                    else:
                        f.write(str(x) + ',')
            f.write('\r')
