#import glob
#import os
#import tensorflow as tf
import numpy as np

print('not allocated')
ar = np.zeros((1000, 1000, 1000), dtype=int)
print('allocated')

"""
loc = "./data/Google/public_data/input_data/task1_v4/dataset_1/train"
abs_loc = os.path.abspath(loc)
path_to_shards = glob.glob(os.path.join(abs_loc, 'shard_*.tfrecord'))
dataset = tf.data.TFRecordDataset(path_to_shards)

"""