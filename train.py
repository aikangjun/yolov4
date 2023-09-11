import os
import tensorflow as tf
from yolov4_model import Yolov4_model
from configure import config as cfg
from _utils.generate import Generator
from _utils.utils import WarmUpCosineDecayScheduler

if __name__ == '__main__':

    yolo_ = Yolov4_model(anchors=cfg.anchors,
                         classes_name=cfg.classes_name,
                         learning_rate=cfg.learning_rate,
                         ignore_thresh=0.5,  # 在计算loss时使用，值越大，负样本个数越少
                         score_thresh=0.3,  # 值越大，产生的框类别越准确
                         iou_thresh=0.3,  # 0.3——0.5值越小，产生的框越少 score_thresh,iou_thresh两个参数都是在生成预测图片时使用
                         max_boxes=cfg.max_boxes)

    data_gen = Generator(annotation_path=cfg.data_dir + '\\annotation.txt',
                         input_size=cfg.input_size,
                         batch_size=cfg.batch_size,
                         train_ratio=cfg.train_ratio,
                         anchors=cfg.anchors,
                         max_boxes=cfg.max_boxes,
                         num_class=cfg.num_classes)

    train_gen = data_gen.generate(training=True)
    valid_gen = data_gen.generate(training=False)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(network=yolo_.network, optimizer=yolo_.optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=3)

    # 如果检查点存在，则恢复最新的检查点，加载模型
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

    if cfg.cosine_scheduler:
        total_steps = data_gen.get_train_step() * cfg.Epoches
        warmup_steps = int(data_gen.get_train_step() * cfg.Epoches * 0.2)
        hold_steps = data_gen.get_train_step() * data_gen.batch_size
        reduce_lr = WarmUpCosineDecayScheduler(global_interval_steps=total_steps,
                                               warmup_interval_steps=warmup_steps,
                                               hold_interval_steps=hold_steps,
                                               learning_rate_base=cfg.learning_rate,
                                               warmup_learning_rate=cfg.warmup_learning_rate,
                                               min_learning_rate=cfg.min_learning_rate,
                                               verbose=0)
    for epoch in range(cfg.Epoches):
        # ----training----
        print('------start training------')
        for i in range(data_gen.get_train_step()):
            sources, targets = next(train_gen)
            if cfg.cosine_scheduler:
                learning_rate = reduce_lr.batch_begin()
                yolo_.optimizer.learning_rate = learning_rate
            yolo_.train(sources, targets)
            if tf.reduce_any([tf.math.is_nan(tf.reduce_sum(varable)) for varable in yolo_.network.trainable_variables]):
                raise Exception('可训练参数为nan')

            if not (i + 1) % 10:
                yolo_.generate_sample(sources, i + 1, letterbox_image=False)
                print('yolo_loss: {}\t'.format(yolo_.train_loss.result().numpy()),
                      'conf_acc: {}\t'.format(yolo_.train_conf.result().numpy() * 100),
                      'class_acc: {}\n'.format(yolo_.train_class_acc.result().numpy() * 100))
        ckpt_save_path = ckpt_manager.save()

        # ----validating----
        print('------start validating------')
        for i in range(data_gen.get_val_step()):
            sources, targets = next(valid_gen)
            yolo_.validata(sources, targets)
            if not (i + 1) % 10:
                print('yolo_loss: {}\t'.format(yolo_.valid_loss.result().numpy()),
                      'conf_acc: {}\t'.format(yolo_.valid_conf.result().numpy() * 100),
                      'class_acc: {}\n'.format(yolo_.valid_class_acc.result().numpy() * 100))

        print(f'Epoch {epoch + 1}\n',
              f'train_yolo_loss: {yolo_.train_loss.result().numpy()}\n',
              f'train_conf_acc: {yolo_.train_conf.result().numpy() * 100}\n',
              f'train_class_acc: {yolo_.train_class_acc.result().numpy() * 100}\n',
              f'valid_yolo_loss: {yolo_.valid_loss.result().numpy()}\n',
              f'valid_conf_acc: {yolo_.valid_conf.result().numpy() * 100}\n',
              f'valid_class_acc: {yolo_.valid_class_acc.result().numpy() * 100}\n')

        yolo_.train_loss.reset_states()
        yolo_.valid_loss.reset_states()

        yolo_.train_conf.reset_states()
        yolo_.valid_conf.reset_states()

        yolo_.train_class_acc.reset_states()
        yolo_.valid_class_acc.reset_states()
