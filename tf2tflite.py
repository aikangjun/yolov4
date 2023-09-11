# tensorflow PB模型转为tflite,在移动端，嵌入式，物联网上使用
# 基于代码的方式
import tensorflow as tf

pb_dir_path = r'.\\tf_model\\'  # pb文件夹的地址
tflite_file_path = r'.\\tflite_model\\yolov4.tflite'  # tflite文件的地址

converter = tf.lite.TFLiteConverter.from_saved_model(pb_dir_path)
# 训练后量化：转为tflite_model后，可以进行训练后量化
# 默认，动态范围量化，权重从浮点静态量化为整数，8位精度，输出是float32
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# float16量化：量化为float16缩减浮点规模大小
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
with open(tflite_file_path, 'wb') as f:
    f.write(tflite_quant_model)
