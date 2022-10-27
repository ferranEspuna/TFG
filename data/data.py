import json
import os
from collections import OrderedDict
from typing import Generator, Tuple
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.keras import Sequential
import h5py
from tensorflow.python.layers.base import Layer
from data.Google.GoogleDatasetReader import load_google_dataset

FOLDER_TEMPLATE_TASK_1 = "./data/Google/public_data/input_data/task1_v4/{}"
DEFAULT_EXAMPLES = 100


def get_data_test() -> np.ndarray:
    num_neurons = 200
    num_examples = 200
    M = np.random.random((num_neurons, num_examples))
    return M


# wtf
def cast_to_integer_if_possible(dct):
    dct = dict(dct)
    for k, v in dct.items():
        if isinstance(v, float) and v.is_integer():
            dct[k] = int(v)
    return dct


# wtf
def wrap_layer(layer_cls, *args, **kwargs) -> Tuple[Layer, str]:
    name = kwargs['layer_name']
    del kwargs['layer_name']

    class wrapped_layer(layer_cls):
        def __call__(self, x, *args, **kwargs):
            self._last_seen_input = x
            return super(wrapped_layer, self).__call__(x, *args, **kwargs)

    return wrapped_layer(*args, **kwargs), name


# wtf
def parse_layer(layer_def):
    layer_cls = getattr(tf.keras.layers, layer_def['layer_name'])
    kwargs = dict(layer_def)
    return wrap_layer(layer_cls, **cast_to_integer_if_possible(kwargs))


# get all the layer names (that have weights) from hdf5 file
def get_layer_names_from_file(location):
    with h5py.File(location, mode='r') as f:
        keys3 = [(key, key2) for key in f for key2 in f[key]]
        return [x for key, key2 in keys3 for x in f[key][key2]]


# lots of tears shed here
def calculate_all_activations_layer_by_layer(x, config_path, weights_path, num_skipped_layers_from_start=1,
                                             skip_reduction_layers=False):

    activations = []
    layer_names_file = get_layer_names_from_file(weights_path)

    with open(config_path, 'r') as f:
        model_def = json.load(f, object_pairs_hook=OrderedDict)

    # to keep track of how many times we have loaded weights
    j = 0

    for i, layer_def in enumerate(model_def['model_config']):

        # construct each layer
        layer, name = parse_layer(layer_def)
        m_layer = Sequential([layer])
        m_layer.build([0] + list(x.shape[1:]))
        has_weights = len(layer.get_weights()) > 0

        # if weights need loading:
        if has_weights:
            # setting the layer name before adding to Sequential won't do!
            m_layer.layers[0]._name = layer_names_file[j]
            j += 1
            m_layer.load_weights(weights_path, by_name=True)

        # add the activations to the dataset. Maybe look into if it's worth it to do with layers with no weights???
        x = m_layer.predict_on_batch(x)

        if i >= num_skipped_layers_from_start:
            if not skip_reduction_layers or has_weights:
                examples_x_neurons = np.reshape(np.copy(x), newshape=(-1, x.shape[0]))
                activations.append(examples_x_neurons)

        tf.keras.backend.clear_session()

    final_array = np.concatenate(activations, axis=0)
    return final_array


def load_all_activations_at_once(x, config_path, weights_path, num_skipped_layers_from_start=1,
                                 skip_reduction_layers=False):

    activations = []
    skipped_iterations = 0

    with open(config_path, 'r') as f:
        config = json.load(f)

    model = Sequential([parse_layer(l) for l in config['model_config']])
    model.build([0] + config['input_shape'])
    model.load_weights(weights_path)

    for layer in model.layers:

        m_layer = tf.keras.Sequential([layer])
        x = m_layer.predict_on_batch(x)

        if skipped_iterations < num_skipped_layers_from_start:
            skipped_iterations += 1
        else:
            if not skip_reduction_layers or len(layer.get_weights() > 0):

                examples_x_neurons = np.reshape(np.copy(x), newshape=(-1, x.shape[0]))
                activations.append(examples_x_neurons)

    final_array = np.concatenate(activations, axis=0)

    return final_array


def get_google_examples(nExamples: int = DEFAULT_EXAMPLES, layer_by_layer: bool = True, skip_reduction: bool = True
                        ) -> Generator[np.ndarray, None, None]:

    # build matrix with some examples
    dataset_location = FOLDER_TEMPLATE_TASK_1.format('dataset_1')
    train_dataset, test_dataset = load_google_dataset(dataset_location)
    npiterator = tfds.as_numpy(train_dataset.take(nExamples))
    x_train_list, _ = zip(*npiterator)
    x_train_list = list(map(lambda x_example: x_example[np.newaxis, ...], x_train_list))
    x_train = np.concatenate(x_train_list, axis=0)

    for i in range(800):

        dirname = 'model_' + str(i)
        model_location = FOLDER_TEMPLATE_TASK_1.format(dirname)
        config_path = os.path.join(model_location, 'config.json')

        if os.path.isdir(model_location):

            for trained in [True, False]:

                if trained:
                    weights_path = os.path.join(model_location, 'weights.hdf5')
                else:
                    weights_path = os.path.join(model_location, 'weights_init.hdf5')

                if layer_by_layer:
                    yield calculate_all_activations_layer_by_layer(x_train, config_path, weights_path,
                                                                   num_skipped_layers_from_start=1,
                                                                   skip_reduction_layers=skip_reduction), \
                          dirname + '_' + str(trained)
                else:
                    yield calculate_all_activations_layer_by_layer(x_train, config_path, weights_path,
                                                                   num_skipped_layers_from_start=1,
                                                                   skip_reduction_layers=skip_reduction), \
                          dirname + '_' + str(trained)
