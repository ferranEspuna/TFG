import glob
import os
import tensorflow as tf

loc = "./data/Google/public_data/input_data/task1_v4/dataset_1/trainihgiuh"
abs_loc = os.path.abspath(loc)
path_to_shards = glob.glob(os.path.join(abs_loc, 'shard_*.tfrecord'))
print(path_to_shards)
dataset = tf.data.TFRecordDataset(path_to_shards)