import torch
print(torch.cuda.is_available())  # True 表示 CUDA 已启用
print(torch.cuda.get_device_name(0))  # 打

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))  # 显示 GPU 设备

