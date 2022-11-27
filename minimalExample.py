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


print(1)
loc = "./data/Google/public_data/input_data/task1_v4/dataset_1/train"
print(2)
abs_loc = os.path.abspath(loc)
print(3)
path_to_shards = glob.glob(os.path.join(abs_loc, 'shard_*.tfrecord'))
print(4)
dataset = tf.data.TFRecordDataset(path_to_shards)
print(5)
deserialized_dataset = dataset.map(_deserialize_example)
print(6)
