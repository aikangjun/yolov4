import tensorflow as tf


# 调整模型输出变成，模型输出为(m,13,13,24),(m,26,26,24),(m,52,52,24)和targets的数据格式不匹配
# 调整为(m,13,13,3,8),(m,26,26,3,8),(m,52,52,3,5+3)
def yolo_head(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = tf.reshape(tf.cast(anchors, tf.float32), (1, 1, 1, num_anchors, 2))

    # 获得x，y的网格
    grid_shape = tf.shape(feats)[1:3]
    x, y = tf.meshgrid(tf.range(grid_shape[1]), tf.range(grid_shape[0]))
    grid_xy = tf.stack([x, y], axis=-1)
    grid = grid_xy[..., tf.newaxis, :]
    grid = tf.cast(grid, dtype=feats.dtype)
    feats = tf.reshape(feats, shape=(-1, grid_shape[1], grid_shape[0], num_anchors, 5 + num_classes))
    # 将预测值调整为真实值
    # box_xy对应框的中心点，
    # sigmoid(xy)表示根据输出层映射出来框的中心离所在格子左上角的相对坐标
    # box_wh对应框的宽和高，
    # anchors_tensor*exp(hw)表示这个预测的框在416*416这个新图片的hw
    # xy除以这个特征图的大小，而wh除以新图片的尺寸，将box_xy,box_wh都缩小到1*1的尺寸
    box_xy = (tf.sigmoid(feats[..., 0:2]) + grid) / tf.cast(grid_shape[..., ::-1], dtype=feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., ::-1], dtype=feats.dtype)

    # 将box_confidence和box_class_probs都缩放到0——1之间，用于计算box_scores
    box_confidence = tf.sigmoid(feats[..., 4:5])
    box_class_probs = tf.nn.softmax(feats[..., 5:], axis=-1)
    return box_xy, box_wh, box_confidence, box_class_probs


# 如果使用letter_box，则对box进行调整，使其符合真实图片的样子
def yolo_correct_box(box_xy, box_wh, input_shape, image_shape):
    # 将y轴放前面，方便预测框与图像的宽高进行相乘
    # 因为image_shape 内保存值的顺序是 [h,w]
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = tf.cast(input_shape, dtype=box_yx.dtype)
    image_shape = tf.cast(image_shape, dtype=box_yx.dtype)

    # new_shape表示缩放后图片的形状
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

    # 求出来的offset是图像有效区域相对于图像左上角的偏移情况,缩放到(0,1)之间
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    # 将范围在(0,1)的坐标y,x值，放大到(0,image_shape[0])和(0,image_shape[1])
    # 对应到真实图片上的坐标
    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    return boxes


# 获取一个特征层的每个box，和它的confidence,class_probs
def get_boxes_and_score(feats, anchors, num_classes, input_shape, image_shape, letterbox_image=True):
    '''

    :param feats:
    :param anchors:
    :param num_classes:
    :param input_shape:
    :param image_shape:
    :param letterbox_image:
    :return: (-1,4)(-1)(-1,num_classes)
    '''
    # 将feats进行调整
    # box_xy  (b,feats_w,feats_h,3,2)
    # box_wh  (b,feats_w,feats_h,3,2)
    # box_confidence  (b,feats_w,feats_h,3,1)
    # box_class_probs  (b,feats_w,feats_h,3,num_classes)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)

    # 在训练阶段，图像传入网络预测会进行letterbox_image,在图像周围添加灰条
    # 生成的box_xy,box_wh都是相对有灰条的图像
    # 需要进行对齐进行修改，去除灰条部分
    # 将box_xy和box_wh 调节为y_min,x_min,y_max,x_max
    # 调用adjust_box()函数进行调整
    if letterbox_image:
        boxes = yolo_correct_box(box_xy, box_wh, input_shape, image_shape)
    # 在推理阶段，直接resize成为(416,416),则进行下面操作
    else:
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        image_shape = tf.cast(image_shape, dtype=box_yx.dtype)

        boxes = tf.concat([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ], axis=-1)

    boxes = tf.reshape(boxes, shape=(-1, 4))
    box_confidence = tf.reshape(box_confidence, shape=(-1))
    box_class_probs = tf.reshape(box_class_probs, shape=(-1, num_classes))
    return boxes, box_confidence, box_class_probs


# 图片预测，输出的每一层feats进行预测，在前面我们已经实现了对一层计算boxes,box_confidence,box_class_probs的函数
def yolo_eval(yolo_outputs: list,
              anchors,
              num_classes,
              image_shape,
              max_boxes,
              score_threshold,
              iou_threshold,
              letterbox_image: bool = False):
    '''

    :param yolo_outputs:
    :param anchors:
    :param num_classes:
    :param image_shape:
    :param max_boxes:
    :param confidence_threshold:
    :param iou_threshold:
    :param letterbox_image: 默认为True,表示使用了letterbox填充方法,
    :return:
    '''
    # 获取yolo输出的特征图数量，3
    num_layers = len(yolo_outputs)
    # 输出特征类别顺序是13*13，26*26，52*52，所以anchor_mask如下
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    batch_size = tf.shape(yolo_outputs[0])[0]
    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32  # 13*32=416
    total_boxes = []
    total_scores = []
    total_classes = []
    for i in range(batch_size):
        # 遍历批量
        boxes = []
        box_confidence = []
        box_class_probs = []
        for l in range(num_layers):
            # 对每个特征层进行处理
            _boxes, _box_confidence, _box_class_probs = get_boxes_and_score(feats=yolo_outputs[l][i][tf.newaxis, ...],
                                                                            anchors=anchors[anchor_mask[l]],
                                                                            num_classes=num_classes,
                                                                            input_shape=input_shape,
                                                                            image_shape=image_shape,
                                                                            letterbox_image=letterbox_image)
            boxes.append(_boxes)
            box_confidence.append(_box_confidence)
            box_class_probs.append(_box_class_probs)
        # 对每个特征层的结果进行堆叠
        boxes = tf.concat(boxes, axis=0)  # (-1,4)
        box_confidence = tf.concat(box_confidence, axis=0)  # (-1,1)
        box_class_probs = tf.concat(box_class_probs, axis=0)  # (-1,num_classes)

        # 判断box_score是否大于分数阈值
        box_scores = box_confidence[..., tf.newaxis] * box_class_probs
        mask = box_scores >= score_threshold

        # masked_boxes = tf.boolean_mask(boxes, mask)
        # masked_scores = tf.boolean_mask(box_scores, mask)
        # masked_probs = tf.boolean_mask(box_class_probs, mask)
        max_boxes_tensor = tf.cast(max_boxes, dtype=tf.int32)  # 最多的boxes数定义

        boxes_ = list()
        score_ = list()
        classes_ = list()
        for c in range(num_classes):
            # 按类别遍历
            # 取出box_scores >= scores_threshold所有的对应类的boxes,和所有对应类的score
            class_boxes = tf.boolean_mask(boxes, mask=mask[..., c])
            class_box_scores = tf.boolean_mask(box_scores[..., c], mask=mask[..., c])

            # 非极大抑制
            # boxes 必须是 (y1,x1,y2,x2)的形式
            # max_output_size 的参数max_box_tensor表示一类中允许有最多box的数量
            nms_index = tf.image.non_max_suppression(boxes=class_boxes,
                                                     scores=class_box_scores,
                                                     max_output_size=max_boxes_tensor,
                                                     iou_threshold=iou_threshold)
            # 注意 tf可以像numpy一样的切片操作，如[:],[:,::-1],[...,-1]
            # 但是不能像numpy能够通过数组（如list）进行索引操作,
            # example:
            # import numpy as np
            # a = np.arange(10)
            # l = [0,5,7]
            # a_ = a[l]
            # print(a_) # [0,5,7]
            # 但要注意数组l的元素内容不能大于a的索引范围

            # tf要实现上述numpy的相似的数组索引操作，必须通过tf.gather()函数，
            # tf.gather()是从目标矩阵中获取需要的数据

            class_boxes = tf.gather(class_boxes, indices=nms_index)
            class_box_scores = tf.gather(class_box_scores, indices=nms_index)
            classes = tf.ones_like(class_box_scores, dtype=tf.int32) * c

            boxes_.append(class_boxes)
            score_.append(class_box_scores)
            classes_.append(classes)

        # 得到每一类通过非极大值抑制后的框后，得到的boxes_，score_，classes_都是一个长度为num_classes的list,需要进行concat
        boxes_ = tf.concat(boxes_, axis=0)
        score_ = tf.concat(score_, axis=0)
        classes_ = tf.concat(classes_, axis=0)  # 进行concat，得到一张图片的boxes_,score_,classes_

        total_boxes.append(boxes_)
        total_scores.append(score_)
        total_classes.append(classes_)  # 加入到列表中，作为batch张图片总的预测框信息、

    return total_boxes, total_scores, total_classes
