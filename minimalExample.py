import glob
import os
import tensorflow as tf


def _deserialize_example(serialized_example):
    record = tf.io.parse_single_example(
        serialized_example,
        features={
            'inputs': tf.io.FixedLenFeature([], tf.string),
            'output': tf.io.FixedLenFeature([], tf.string)
        })
    inputs = tf.io.parse_tensor(record['inputs'], out_type=tf.float32)
    output = tf.io.parse_tensor(record['output'], out_type=tf.int32)
    return inputs, output


loc = "./data/Google/public_data/input_data/task1_v4/dataset_1/train"
abs_loc = os.path.abspath(loc)
path_to_shards = glob.glob(os.path.join(abs_loc, 'shard_*.tfrecord'))
dataset = tf.data.TFRecordDataset(path_to_shards)
deserialized_dataset = dataset.map(_deserialize_example)
