import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

from xyolo import DefaultYolo3Config, YOLO
from xyolo import init_yolo_v3
import os

project_path = os.path.join('.')
data_path = os.path.join('.', 'data', 'tfrecord')

# 
class MyConfig(DefaultYolo3Config):
    def __init__(self):
        super(MyConfig, self).__init__()
        self.classes_path = os.path.join(project_path, 'classes.txt')
        self.tf_train_path = os.path.join(data_path, 'train_tf')
        self.tf_val_path = os.path.join(data_path, 'val_tf')
        self.frozen_batch_size = 64
        self.unfreeze_batch_size = 32
        self._output_model_path = os.path.join(project_path, 'test_xyolo','output_model.h5')

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

config = MyConfig()
init_yolo_v3(config)
yolo = YOLO(config=config, train=True)
yolo.fit()