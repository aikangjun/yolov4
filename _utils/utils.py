import cv2
import numpy as np
from PIL import Image


def get_anchors(anchors_path: str):
    '''从文件中加载anchors'''
    with open(anchors_path) as f:
        line = f.readline()
    anchors = []
    for x in line.split('\t'):
        w_h = x.split(',')
        anchors.append([int(w_h[0]), int(w_h[1])])
    return np.array(anchors)


def rand(a=0, b=1):
    '''产生随机数范围为(a,b)之间'''
    return np.random.rand() * (b - a) + a


def letterbox_image(image: Image, size: tuple):
    '''
    letterbox_image一种图像前处理的方式。像信封一样，图像保持长宽比的情况下，
    填充到一个盒子内，长边缩放到size[0]长度,短边用黑色或灰色填充，在这里用灰色填充
    :param image: 输入的图片
    :param size:需要resize到的形状
    :return:
    '''
    iw, ih = image.size
    w, h = size
    # 转换比例
    scale = min(w / iw, h / ih)
    # nw,nh为转换后的图像尺寸，进行同比例缩放操作
    # 将其中一个长度转为了416，另外一个长度空缺部分，用灰色填充像素
    nw = int(iw * scale)
    nh = int(ih * scale)
    # 这里拿到dx,dy是为了得到paste()需要的左上角坐标
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    # 对图片进行resize(),重采样方式为Image.BICUBIC双三次插值
    # NEAREST BOX BILINEAR HAMMING BICUBIC LANCZOS
    image = image.resize(size=(nw, nh), resample=Image.BICUBIC)
    # 创建一个416*416 的 RGB 图像，像素全为灰色
    new_image = Image.new(mode='RGB', size=(w, h),
                          color=(128, 128, 128))
    # paste() 将另一个图片粘贴到此图片
    new_image.paste(im=image, box=(dx, dy))
    return new_image, scale, dx, dy


def get_random_data(line: str,
                    image_size: tuple,
                    max_boxes: int,
                    random: bool = True,
                    jitter: float = .3,
                    hue: float = .1,
                    sat: float = 1.5,
                    val: float = 1.5):
    '''
    使用随机预处理进行实时数据争强,对image和box都进行了缩放操作,image数据进行归一化，box数据没有归一化操作
    :param line: 从txt文件读取的str信息，使用'\t'隔开,包含图片路径和box信息
    :param image_size: 输入网络的图片大小
    :param random: 是否进行随机数据增强
    :param max_boxes: 指定最大目标框的个数
    :param jitter: 抖动，用于rand()函数，生成随机数 生成随机高宽比
    :param hue:  hue色相
    :param sat:  saturation饱和度
    :param val:  value色明度
    :param proc_img: 是否进行数据预处理
    :return image_data：经过处理后的的图像数据,添加了灰条,进行了缩放到(416,416)的操作，和像素归一化到[0,1]
    box_data:形状为(max_boxes,5),将原始bounding box同比例缩放到(416，416)
    '''
    line_list = line.split(' ')
    image = Image.open(line_list[0])
    iw, ih = image.size  # 输入图片的原始尺寸 例如320，500
    h, w = image_size  # 转换的尺寸 416，416

    # [[   248     80     19     76     44   1444]
    #  [   291     44     31     61     44   1891]
    #  [   184    134     31     88     44   2728]
    #  [   341    156     33    105     44   3465]
    #  [   445    240     71     50     50   3550]
    #  [   449    229     72     56     50   4032]
    #  [   238    235     77     70     58   5390]
    #  [    18    175     34    175     47   5950]
    #  [    47     81     93    160      1  14880]
    #  [   182    341    359    156     67  56004]
    #  [   492    172    292    343      1 100156]]
    # box的数据形式
    box = np.array(
        [np.array(list(map(int, box.split(',')))) for box in line_list[1:-1]])

    if not random:
        # 如果不随机变换原始数据
        new_image, scale, dx, dy = letterbox_image(image, image_size)
        image_data = np.array(new_image, dtype=np.float32) / 255
        # clip()现在数组中的值
        image_data = np.clip(image_data, 0., 1.)

        # correct boxes
        # 建立(max_boxes,5)的全零矩阵，用于存centter_point_x,
        # center_point_y,width,height,category_id
        box_data = np.zeros(shape=(max_boxes, 5))
        if box.__len__() > 0:
            # 如果图片中设置的box,进行操作
            np.random.shuffle(box)  # 将box随机打乱
            if box.__len__() > max_boxes:
                # box的数量大于设置的max_boxes,只取max_boxes个
                box = box[:max_boxes]
            # 对目标框进行同比例缩放和位移,相对于new_image
            box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
            box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
            box_data[:len(box)] = box
        return image_data, box_data
    if random:
        # 如果进行随机数据增强，则要随机盖伯安图片尺寸和y颜色空间的取值范围
        # 1.resize image对图像进行缩放
        # 随机生成 宽高比
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter,
                                                             1 + jitter)
        # 随机生成缩放比例
        scale = rand(.25, 2)
        # 计算新图片尺寸
        if new_ar < 1:  # w<h
            nh = int(ih * scale)
            nw = int(iw * new_ar)
        else:  # w>h
            nw = int(iw * new_ar)
            nh = int(ih * scale)
        image = image.resize(size=(nw, nh), resample=Image.BICUBIC)
        # 2.平移变换，随机的把变换后的图片放置在灰度图像上，随机水平位移 place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))  # 得到随机的paste()需要的左上角坐标
        new_image = Image.new(mode='RGB', size=(w, h), color=(128, 128, 128))
        new_image.paste(im=image, box=(dx, dy))
        image = new_image
        # 3.翻转 是否翻转图像 flip image or not
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 4.图片色域扭曲 distort image
        # 在hsv
        hue = rand(-hue, hue)  # 得到一个-0.1到0.1的色相
        # 0.5的概率得到[1,1.5]的一个饱和度，
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        image = np.array(image, np.float32) / 255
        image = np.clip(image, 0., 1.)
        x = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2HSV)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image_date = cv2.cvtColor(src=x,
                                  code=cv2.COLOR_RGB2HSV)  # numpy array 0 to 1

        # correct box
        # 经过缩放和平移后，box信息需要做同等变化
        box_data = np.zeros(shape=(max_boxes, 5))
        if box.__len__() > 0:
            np.random.shuffle(box)
            # 对目标框进行同比例缩放和位移,相对于new_image
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:[2, 0]]
            # 如果center_point坐标小于0，将坐标置为0
            box[:, [0, 1]][box[:, [0, 1]] < 0] = 0

            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            # 丢弃无效框
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            if box.__len__() > max_boxes:
                # box的数量大于设置的max_boxes,只取max_boxes个
                box = box[:max_boxes]
            box_data[:max_boxes] = box
        return image_date, box_data


def preprocess_true_boxes(true_boxes: np.ndarray,
                          input_shape: tuple,
                          anchors: np.ndarray,
                          num_classes: int):
    '''
    预处理真实框,将左上角和右下角坐标转为中心点坐标和宽高,
    并将真实框归一化到[0,1]之间，方便计算iou
    :param true_boxes: true_boxes的形状，(batch_size,max_boxes,5)
    :param input_shape: (416,416)
    :param anchors: (9,2)
    :param num_classes: 3
    :return:
    '''
    # assert Flase,'可选信息'    当断言为False时，打印可选信息
    assert (true_boxes[..., 4] < num_classes).all(), 'class_i必须小于num_class'
    num_layers = len(anchors) // 3  # num_layers

    # 13*13,26*26,52*52特征层对应的anchor均有3个
    # anchor是对预测的对象范围进行约束，加入尺寸先验条件，实现对尺度学习的目的
    # 13*13 的特征图感受野最大,使用最大的3个anchor
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # 得到框的坐标的图片大小
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='float32')

    # 计算获得中心点坐标和宽高 形状均为(bacth_size,max_boxes,2)
    true_boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    true_boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    # 注意:这是的true_box里存的时中心点坐标和高宽，框中的数据压缩到[0,1]
    # 这样处理方便映射到gird中，便于将真实框调整到13*13，26*26，52*52之间
    true_boxes[..., 0:2] = true_boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = true_boxes_wh / input_shape[::-1]
    # (x,y,w,h)
    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in
                   range(num_layers)]  # (3,2) [[13,13],[26,26],[52,52]]

    y_true = [np.zeros(shape=(m, int(grid_shapes[l][0]), int(grid_shapes[l][1]), len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # y_true 是一个list,内部数据的形状为(m,13,13,3,c)(m,26,26,3,c)(m,52,52,3,c)  c=4+1+num_classes

    # (9,2) -> (1,9,2)
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    # true_boxes_wh中不为0的有效数据的bool值, 形状为(batch_size,max_boxes)
    valid_mask = true_boxes_wh[..., 0] > 0

    for b in range(m):
        # 对每张图进行处理,在(416,416)的尺寸进行处理
        wh = true_boxes_wh[b, valid_mask[b]]  # (n,2)
        if len(wh) == 0:
            continue
        # (n,2) -> (n,1,2)
        wh = np.expand_dims(wh, axis=-2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        # 计算真实框和先验框的交并比
        # interscet_area (n,9)
        # box_area (n,1)
        # abchor_area (1,9)
        # iou (n,9)
        # best_iou (n,)
        intersect_mins = np.maximum(box_mins, anchor_mins)  # (n,1,2)(1,9,2) ->(n,9,2)每个真实框都与anchor比较
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)  # 每个真实框与anchors的交并比
        best_anchor = np.argmax(iou, axis=-1)  # (n,)找到一张土拍你每个框对应的最合适的anchor的索引
        for t, n in enumerate(best_anchor):
            # -----------------------------------------------------------#
            #   找到每个真实框所属的特征层
            # -----------------------------------------------------------#
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # -----------------------------------------------------------#
                    #   true_boxes的中心点和宽高为[0,1],下面操作将中心点放在到grid的坐标，i,j表示对应在grid的x,y
                    # -----------------------------------------------------------#
                    i = np.floor(
                        true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(
                        true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    # -----------------------------------------------------------#
                    #   k指的的当前这个特征点的第k个先验框
                    # -----------------------------------------------------------#
                    k = anchor_mask[l].index(n)
                    # -----------------------------------------------------------#
                    #   c指的是当前这个真实框的种类
                    # -----------------------------------------------------------#
                    c = true_boxes[b, t, 4].astype('int32')
                    # -----------------------------------------------------------#
                    #   y_true的shape为(m,13,13,3,c)(m,26,26,3,c)(m,52,52,3,c)
                    #   最后的c可以拆分为框的中心、宽高、置信度以及类别概率
                    # -----------------------------------------------------------#
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


def cosine_decay_with_warmup(global_step,
                             total_steps,
                             warmup_steps,
                             hold_steps,
                             learning_rate_base,
                             warmup_learning_rate,
                             min_learning_rate):
    if any([learning_rate_base, warmup_learning_rate, min_learning_rate]) < 0:
        raise ValueError('all of the learning rates must be greater than 0.')

    if np.logical_or(total_steps < warmup_steps, total_steps < hold_steps):
        raise ValueError('total_steps must be larger or equal to the other steps.')

    if np.logical_or(learning_rate_base < min_learning_rate, warmup_learning_rate < min_learning_rate):
        raise ValueError('learning_rate_base and warmup_learning_rate must be larger or equal to min_learning_rate.')

    if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')

    if global_step < warmup_steps:

        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        return slope * global_step + warmup_learning_rate

    elif warmup_steps <= global_step <= warmup_steps + hold_steps:

        return learning_rate_base

    else:
        return 0.5 * learning_rate_base * (1 + np.cos(np.pi * (global_step - warmup_steps - hold_steps) /
                                                      (total_steps - warmup_steps - hold_steps)))


class WarmUpCosineDecayScheduler:

    def __init__(self,
                 global_step=0,
                 global_step_init=0,
                 global_interval_steps=None,
                 warmup_interval_steps=None,
                 hold_interval_steps=None,
                 learning_rate_base=None,
                 warmup_learning_rate=None,
                 min_learning_rate=None,
                 interval_epoch=[0.05, 0.15, 0.3, 0.5],
                 verbose=None,
                 **kwargs):
        self.global_step = global_step
        self.global_steps_for_interval = global_step_init
        self.global_interval_steps = global_interval_steps
        self.warmup_interval_steps = warmup_interval_steps
        self.hold_interval_steps = hold_interval_steps
        self.learning_rate_base = learning_rate_base
        self.warmup_learning_rate = warmup_learning_rate
        self.min_learning_rate = min_learning_rate
        self.interval_index = 0
        self.interval_epoch = interval_epoch
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch) - 1):
            self.interval_reset.append(self.interval_epoch[i + 1] - self.interval_epoch[i])
        self.interval_reset.append(1 - self.interval_epoch[-1])
        self.verbose = verbose

    def batch_begin(self):
        if self.global_steps_for_interval in [0] + [int(j * self.global_interval_steps) for j in self.interval_epoch]:
            self.total_steps = int(self.global_interval_steps * self.interval_reset[self.interval_index])
            self.warmup_steps = int(self.warmup_interval_steps * self.interval_reset[self.interval_index])
            self.hold_steps = int(self.hold_interval_steps * self.interval_reset[self.interval_index])
            self.interval_index += 1
            self.global_step = 0

        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      total_steps=self.total_steps,
                                      warmup_steps=self.warmup_steps,
                                      hold_steps=self.hold_steps,
                                      learning_rate_base=self.learning_rate_base,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      min_learning_rate=self.min_learning_rate)

        self.global_step += 1
        self.global_steps_for_interval += 1

        if self.verbose:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_steps_for_interval, lr))

        return lr
