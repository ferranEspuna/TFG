import os
import json
import time
from collections import OrderedDict
from typing import Generator, Tuple, List, Dict, Callable
import numpy as np
import tensorflow_datasets as tfds
import h5py
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.layers.base import Layer
from data.GoogleDatasetReader import load_google_dataset
from threading import Thread

FOLDER_TEMPLATE_TASK_1 = "./data/Google/public_data/input_data/task1_v4/{}"


def get_data_test() -> np.ndarray:
    num_neurons = 200
    num_examples = 200
    m = np.random.random((num_neurons, num_examples))
    return m


# wtf
def cast_to_integer_if_possible(d: Dict) -> Dict:
    d = dict(d)
    for k, v in d.items():
        if isinstance(v, float) and v.is_integer():
            d[k] = int(v)
    return d


# wtf
def wrap_layer(layer_cls, *args, **kwargs) -> Tuple[Layer, str]:
    name = kwargs['layer_name']
    del kwargs['layer_name']

    class wrapped_layer(layer_cls):
        def __call__(self, x, *args1, **kwargs1):
            self._last_seen_input = x
            return super(wrapped_layer, self).__call__(x, *args1, **kwargs1)

    return wrapped_layer(*args, **kwargs), name


# wtf
def parse_layer(layer_def: Dict) -> Tuple[Layer, str]:
    layer_cls = getattr(tf.keras.layers, layer_def['layer_name'])
    kwargs = dict(layer_def)
    return wrap_layer(layer_cls, **cast_to_integer_if_possible(kwargs))


# get all the layer names (that have weights) from hdf5 file
def get_layer_names_from_file(location: str) -> List[str]:
    with h5py.File(location, mode='r') as f:
        keys3 = [(key, key2) for key in f for key2 in f[key]]
        return [x for key, key2 in keys3 for x in f[key][key2]]


def process_layer(d, layer_def, layer_name, weights_path):
    import tensorflow as tf
    # construct each layer
    layer, name = parse_layer(layer_def)
    m_layer = Sequential([layer])
    m_layer.build([0] + list(d['x_train'].shape[1:]))
    has_weights = len(layer.get_weights()) > 0

    # if weights need loading:
    if has_weights:
        # setting the layer name before adding to Sequential won't do!
        m_layer.layers[0]._name = layer_name  # layer_names_file[j]
        m_layer.load_weights(weights_path, by_name=True)

    d['x_train'] = m_layer.predict_on_batch(d['x_train'])
    d['x_test'] = m_layer.predict_on_batch(d['x_test'])
    d['hw'] = has_weights


# lots of tears shed here
def calculate_all_activations_layer_by_layer(x_train: np.ndarray, x_test: np.ndarray, config_path: str,
                                             weights_path: str,
                                             nNeurons: int,
                                             sample_neurons_strategy: Callable[
                                                 [np.ndarray, np.ndarray, int], np.ndarray],
                                             num_skipped_layers_from_start: int = 1, skip_reduction_layers: bool = False
                                             ) -> Tuple[np.ndarray, np.ndarray]:
    activations_train = []
    activations_test = []

    layer_names_file = get_layer_names_from_file(weights_path)

    with open(config_path, 'r') as f:
        model_def = json.load(f, object_pairs_hook=OrderedDict)

    # to keep track of how many times we have loaded weights
    j = 0

    d = {'x_train': x_train, 'x_test': x_test}

    for i, layer_def in enumerate(model_def['model_config']):

        t = Thread(target=process_layer, args=(d, layer_def, layer_names_file[j], weights_path))
        t.start()
        t.join()
        x_train = d['x_train']
        x_test = d['x_test']
        has_weights = d['hw']

        if has_weights:
            j += 1

        if i >= num_skipped_layers_from_start:
            if not skip_reduction_layers or has_weights:
                train_x_neurons = np.reshape(np.copy(x_train), newshape=(-1, x_train.shape[0]))
                activations_train.append(train_x_neurons)

                test_x_neurons = np.reshape(np.copy(x_test), newshape=(-1, x_test.shape[0]))
                activations_test.append(test_x_neurons)

    final_array_train = np.concatenate(activations_train, axis=0)
    final_array_test = np.concatenate(activations_test, axis=0)

    assert final_array_train.shape == final_array_test.shape
    indices = sample_neurons_strategy(final_array_train, final_array_test, nNeurons)
    return final_array_train[indices, :], final_array_test[indices, :]


def calculate_all_activations_at_once(x_train: np.ndarray, x_test: np.ndarray, config_path: str, weights_path: str,
                                      nNeurons: int,
                                      sample_neurons_strategy: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                                      num_skipped_layers_from_start: int = 1, skip_reduction_layers: bool = False
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    activations_train = []
    activations_test = []

    skipped_iterations = 0

    with open(config_path, 'r') as f:
        model_def = json.load(f)

    model = Sequential([parse_layer(lay)[0] for lay in model_def['model_config']])
    model.build([0] + model_def['input_shape'])
    model.load_weights(weights_path)

    for layer in model.layers:

        m_layer = tf.keras.Sequential([layer])
        x_train = m_layer.predict_on_batch(x_train)
        x_test = m_layer.predict_on_batch(x_test)

        if skipped_iterations < num_skipped_layers_from_start:
            skipped_iterations += 1
        else:
            if not skip_reduction_layers or len(layer.get_weights()) > 0:
                train_x_neurons = np.reshape(np.copy(x_train), newshape=(-1, x_train.shape[0]))
                activations_train.append(train_x_neurons)
                print(len(activations_train))

                test_x_neurons = np.reshape(np.copy(x_test), newshape=(-1, x_test.shape[0]))
                activations_test.append(test_x_neurons)
                print(len(activations_test))

    final_array_train = np.concatenate(activations_train, axis=0)
    final_array_test = np.concatenate(activations_test, axis=0)

    assert final_array_train.shape == final_array_test.shape
    indices = sample_neurons_strategy(final_array_train, final_array_test, nNeurons)
    return final_array_train[:, indices], final_array_test[:, indices]


def calculate_activations_by_batches(x_train: np.ndarray, x_test: np.ndarray, config_path: str, weights_path: str,
                                     nNeurons: int,
                                     sample_neurons_strategy: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                                     num_skipped_layers_from_start: int = 1, skip_reduction_layers: bool = False
                                     ) -> Tuple[np.ndarray, np.ndarray]:

    t0 = time.time()

    with open(config_path, 'r') as f:
        model_def = json.load(f)

    model = Sequential([parse_layer(lay)[0] for lay in model_def['model_config']])
    model.build([0] + model_def['input_shape'])
    model.load_weights(weights_path)
    t1 = time.time()
    print('Loaded model in {:.2f}s'.format(t1-t0))

    t0 = time.time()
    total_examples = x_train.shape[1]
    batch_size = 100

    activations_train_all_samples = []
    activations_test_all_samples = []

    start_index = 0
    sample_indices = None

    while start_index < total_examples:

        activations_train_sample = []
        activations_test_sample = []

        end_index = min(start_index + batch_size, total_examples)
        x_train_sample = x_train[start_index: end_index]
        x_test_sample = x_test[start_index: end_index]

        skipped_iterations = 0

        for layer in model.layers:

            m_layer = tf.keras.Sequential([layer])
            x_train_sample = m_layer.predict_on_batch(x_train_sample)
            x_test_sample = m_layer.predict_on_batch(x_test_sample)

            if skipped_iterations < num_skipped_layers_from_start:
                skipped_iterations += 1
            else:
                if not skip_reduction_layers or len(layer.get_weights()) > 0:
                    train_x_neurons_sample = np.reshape(np.copy(x_train_sample), newshape=(-1, x_train_sample.shape[0]))
                    activations_train_sample.append(train_x_neurons_sample)

                    test_x_neurons_sample = np.reshape(np.copy(x_test_sample), newshape=(-1, x_test_sample.shape[0]))
                    activations_test_sample.append(test_x_neurons_sample)

        example_sample_train = np.concatenate(activations_train_sample, axis=0)
        example_sample_test = np.concatenate(activations_test_sample, axis=0)

        if sample_indices is None:
            sample_indices = sample_neurons_strategy(example_sample_train, example_sample_test, nNeurons)

        activations_train_all_samples.append((example_sample_train[sample_indices, :]))
        activations_test_all_samples.append((example_sample_test[sample_indices, :]))

        start_index = end_index
        print(f'Calculated activations for {end_index} / {total_examples} examples')

    final_sample_train = np.concatenate(activations_train_all_samples, axis=1)
    final_sample_test = np.concatenate(activations_test_all_samples, axis=1)

    t1 = time.time()

    print('Calculated activations in {:.2f}s'.format(t1-t0))

    assert final_sample_train.shape == final_sample_test.shape
    return final_sample_train, final_sample_test


def get_x_y_as_matrix(dataset, nExamples):
    npiterator = tfds.as_numpy(dataset.take(nExamples))
    x_list, y_list = zip(*npiterator)

    x_list = list(map(lambda x_example: x_example[np.newaxis, ...], x_list))
    x_train = np.concatenate(x_list, axis=0)

    y_list = list(map(lambda y_example: y_example[np.newaxis, ...], y_list))
    y_train = np.concatenate(y_list, axis=0)

    return x_train, y_train


def get_google_examples(nExamples: int, nNeurons: int,
                        calculate_activations,
                        sample_neurons_strategy: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                        skip_reduction: bool = True
                        ) -> Generator[Tuple[Callable[[], Tuple[np.ndarray, np.ndarray]], str], None, None]:

    t0 = time.time()

    # build matrix with some examples
    dataset_location = FOLDER_TEMPLATE_TASK_1.format('dataset_1')
    train_dataset, test_dataset = load_google_dataset(dataset_location)

    x_train, y_train = get_x_y_as_matrix(train_dataset, nExamples)
    x_test, y_test = get_x_y_as_matrix(test_dataset, nExamples)

    t1 = time.time()
    print('Loaded dataset in {:.2f}s'.format(t1 - t0))

    for i in range(800):

        dirname = 'model_' + str(i)
        model_location = FOLDER_TEMPLATE_TASK_1.format(dirname)
        config_path = os.path.join(model_location, 'config.json')

        if os.path.isdir(model_location):

            for trained in [True]:

                if trained:
                    weights_path = os.path.join(model_location, 'weights.hdf5')
                else:
                    weights_path = os.path.join(model_location, 'weights_init.hdf5')

                def calc_acts():
                    return calculate_activations(x_train, x_test, config_path, weights_path,
                                                 nNeurons=nNeurons, sample_neurons_strategy=sample_neurons_strategy,
                                                 num_skipped_layers_from_start=1,
                                                 skip_reduction_layers=skip_reduction)

                yield calc_acts, dirname + '_' + str(trained)
