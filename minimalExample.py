import glob
import os
import tensorflow as tf

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
print('options')
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
print('session')

loc = "./data/Google/public_data/input_data/task1_v4/dataset_1/train"
abs_loc = os.path.abspath(loc)
path_to_shards = glob.glob(os.path.join(abs_loc, 'shard_*.tfrecord'))
dataset = tf.data.TFRecordDataset(path_to_shards)