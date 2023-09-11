# tf的默认模型是通过pb文件保存
# 构造相同的模型，加载ckpt,再保存为tf模型
import tensorflow as tf
from yolov4_model import Yolov4_model
import configure.config as cfg
import tensorflow.keras as keras

layers = keras.layers
models = keras.models
# 只要两个模型具有相同的架构，它们就可以共享同一个检查点
yolo = Yolov4_model(anchors=cfg.anchors,
                    classes_name=cfg.classes_name,
                    learning_rate=cfg.learning_rate,
                    ignore_thresh=0.5,
                    score_thresh=0.3,
                    iou_thresh=0.3,
                    max_boxes=cfg.max_boxes).network
ckpt = tf.train.Checkpoint(network=yolo)
ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                          directory=cfg.ckpt_path,
                                          max_to_keep=3)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print('Latest checkpoint restored!!')
# 低级API
# yolo_.network.save(filepath='.\\tf_model', save_format='tf')
# 高级API,保存和序列化，函数式模型
# 无状态层不会改变权重，因此即便存在额外的/缺失的无状态层，模型也可以具有兼容架构。
inputs = tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32, name='input')  # 指定模型的输入
yolo._set_inputs(inputs=inputs)
yolo.save(filepath='.\\tf_model\\DETECTION\\1', save_format='tf')
# 在terminal中允许 saved_model_cli show --dir  tf_model/DETECTION/1 --all 显示输入输出签名
