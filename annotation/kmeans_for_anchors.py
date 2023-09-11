import numpy as np
from PIL import Image
import configure.config as cfg


def cas_iou(box, cluster):
    # https://blog.csdn.net/weixin_44791964/article/details/102687531
    '''

    :param box:形状为(2,)有宽高两个维度
    :param cluster: 形状为(k,2)
    :return:
    '''
    # 广播机制
    # np.minimum() 比较两个数组并返回一个包含元素最小值的新数组
    # 将k个聚类框与该box求相交x,y。拿到最小的高和宽
    x = np.minimum(cluster[:, 0], box[0])  # (k,)
    y = np.minimum(cluster[:, 1], box[1])  # (k,)

    intersection = x * y  # (k,)
    area1 = box[0] * box[1]  # (1,)

    area2 = cluster[:, 0] * cluster[:, 1]  # (k,)
    iou = intersection / (area1 + area2 - intersection)
    # 输出形状为 (k,) 与每个anchor的iou
    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box: list, k: int):
    """
    :param box: actual detection frames 形状为(n,2)
    :param k: number of a priori frames (num of sensory fields × num of sizes)
    :return:
    """
    row = box.shape[0]
    # initialize the (1-iou) of all frames corresponding to priori frames
    distance = np.empty(shape=(row, k))
    # initialize the optimal (1-iou) of all frames
    last_clu = np.zeros((row,))  # (row,)

    np.random.seed()
    # 随机在box中选取k个聚类中心 (k,2)
    cluster = box[np.random.choice(row, size=(k,), replace=False)]
    while True:
        for i in range(row):
            # use iou to replace euclidean distance
            distance[i] = 1 - cas_iou(box[i], cluster)
        # distance 形状为(row,k)
        # 取出最小值点
        near = np.argmin(distance, axis=1)  # (row,)
        # loop termination condition
        # .all()判断给定的可迭代参数iterable中的所有元素是否都为 TRUE
        if (last_clu == near).all():
            break

        # update
        # 把near中所有值为j的行所代表得到真实box全找出来，
        # 求box中位数作为新的第j号聚类框，反复迭代直到near不再变化
        # 聚类完成
        for j in range(k):
            # np.median
            cluster[j] = np.median(
                box[near == j], axis=0)

        last_clu = near.copy()

    return cluster


def read_data(lines):
    '''

    :param line: 输入一行annotation文本信息
    :return:scales: 比例
    '''
    line = lines.split('\t')
    h, w = Image.open(line[0]).size
    # [[x1,y1,x2,y2]，[]]
    boxes = np.array([list(map(int, box.split(',')[:-1]))
                      for box in line[1:-1]], dtype='float')

    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h
    # 除以w,h 相当于将坐标全部放到原点位置，所有box从原点出发
    # [[x1/w, y1/h, x2/w, y2/h]，[]]
    scales = boxes[:, [2, 3]] - boxes[:, [0, 1]]
    # scales 为[[(x2-x1)/w, (y2-y1)/h]]
    return scales


if __name__ == '__main__':
    SIZE = cfg.input_size  # (416,416)
    anchors_num = 9

    f = open(file=r'..\data_info\annotation.txt', mode='r')
    lines = f.readlines()
    f.close()
    data = list()
    for line in lines:
        # scales 为[[(x2-x1)/w, (y2-y1)/h]]
        scales = read_data(line)
        # extend()和append()都表示向list添加元素，二者都以容器形式添加。
        # append()添加是将容器看作整体添加，extend()将容器打碎后添加(加的只是元素)
        data.extend(scales.tolist())
    # 将所有 坐标信息 保存到data, 形状为(n,2)
    data = np.array(data)

    # k-means
    # 聚类信息是使用[[(x2-x1)/w, (y2-y1)/h]],真实框宽高和图片宽高的比值
    print('正将进行聚类...')
    out = kmeans(box=data, k=anchors_num)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    data = out * list(reversed(SIZE))  # reversed()返回的是一个迭代器
    print(f'cluster:{data} ')
    f = open(cfg.data_dir + '\\anchors.txt', 'w')
    # anchors是(w,h)二元组，都缩放在0~416内
    for i in range(np.shape(data)[0]):
        if i == 0:
            x_y = "%d,%d" % (round(data[i][0]), round(data[i][1]))
        else:
            x_y = "\t%d,%d" % (round(data[i][0]), round(data[i][1]))
        f.write(x_y)
    f.close()
