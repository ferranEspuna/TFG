import os
import json
import time
from typing import Generator, Tuple, Dict, Callable
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.layers.base import Layer
from data.GoogleDatasetReader import load_google_dataset, load_google_train_dataset

FOLDER_TEMPLATE_TASK_1 = "./Google/public_data/input_data/task1_v4/{}"
BATCH_SIZE = 500


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
    print('Loaded model in {:.2f}s'.format(t1 - t0))

    t0 = time.time()
    total_examples = x_train.shape[0]

    activations_train_all_samples = []
    activations_test_all_samples = []

    start_index = 0
    sample_indices = None

    while start_index < total_examples:

        activations_train_sample = []
        activations_test_sample = []

        end_index = min(start_index + BATCH_SIZE, total_examples)
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

    print('Calculated activations in {:.2f}s'.format(t1 - t0))

    assert final_sample_train.shape == final_sample_test.shape
    return final_sample_train, final_sample_test


def calculate_activations_by_batches_train(x_train: np.ndarray, config_path: str, weights_path: str,
                                           nNeurons: int,
                                           sample_neurons_strategy: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                                           num_skipped_layers_from_start: int = 1, skip_reduction_layers: bool = False
                                           ) -> np.ndarray:
    t0 = time.time()

    with open(config_path, 'r') as f:
        model_def = json.load(f)

    model = Sequential([parse_layer(lay)[0] for lay in model_def['model_config']])
    model.build([0] + model_def['input_shape'])
    model.load_weights(weights_path)
    t1 = time.time()
    print('Loaded model in {:.2f}s'.format(t1 - t0))

    t0 = time.time()
    total_examples = x_train.shape[0]

    activations_train_all_samples = []

    start_index = 0
    sample_indices = None

    while start_index < total_examples:

        activations_train_sample = []

        end_index = min(start_index + BATCH_SIZE, total_examples)
        x_train_sample = x_train[start_index: end_index]

        skipped_iterations = 0

        for layer in model.layers:

            m_layer = tf.keras.Sequential([layer])
            x_train_sample = m_layer.predict_on_batch(x_train_sample)

            if skipped_iterations < num_skipped_layers_from_start:
                skipped_iterations += 1
            else:
                if not skip_reduction_layers or len(layer.get_weights()) > 0:
                    train_x_neurons_sample = np.reshape(np.copy(x_train_sample), newshape=(-1, x_train_sample.shape[0]))
                    activations_train_sample.append(train_x_neurons_sample)

        example_sample_train = np.concatenate(activations_train_sample, axis=0)

        if sample_indices is None:
            sample_indices = sample_neurons_strategy(example_sample_train, example_sample_train, nNeurons)

        activations_train_all_samples.append((example_sample_train[sample_indices, :]))

        start_index = end_index
        print(f'Calculated activations for {end_index} / {total_examples} examples')

    final_sample_train = np.concatenate(activations_train_all_samples, axis=1)
    t1 = time.time()

    print('Calculated activations in {:.2f}s'.format(t1 - t0))
    return final_sample_train


def get_x_y_as_matrix(dataset, nExamples):
    print('shuffle')
    npiterator = tfds.as_numpy(dataset.shuffle(1000000).take(nExamples))
    x_list, y_list = zip(*npiterator)

    x_list = list(map(lambda x_example: x_example[np.newaxis, ...], x_list))
    x_train = np.concatenate(x_list, axis=0)

    y_list = list(map(lambda y_example: y_example[np.newaxis, ...], y_list))
    y_train = np.concatenate(y_list, axis=0)

    return x_train, y_train


def get_google_examples_train(nExamples: int, nNeurons: int,
                              sample_neurons_strategy: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                              skip_reduction: bool = True
                              ) -> Generator[Tuple[Callable[[], np.ndarray], str], None, None]:
    t0 = time.time()

    # build matrix with some examples
    dataset_location = FOLDER_TEMPLATE_TASK_1.format('dataset_1')
    print(dataset_location)
    train_dataset = load_google_train_dataset(dataset_location)
    print('load')

    x_train, y_train = get_x_y_as_matrix(train_dataset, nExamples)

    t1 = time.time()
    print('Loaded {} examples in {:.2f}s'.format(x_train.shape, t1 - t0))

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
                    return calculate_activations_by_batches_train(x_train, config_path, weights_path,
                                                                  nNeurons=nNeurons,
                                                                  sample_neurons_strategy=sample_neurons_strategy,
                                                                  num_skipped_layers_from_start=1,
                                                                  skip_reduction_layers=skip_reduction)

                yield calc_acts, dirname + '_' + str(trained)


def get_google_examples(nExamples: int, nNeurons: int,
                        sample_neurons_strategy: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
                        skip_reduction: bool = True
                        ) -> Generator[Tuple[Callable[[], Tuple[np.ndarray, np.ndarray]], str], None, None]:
    t0 = time.time()

    # build matrix with some examples
    dataset_location = FOLDER_TEMPLATE_TASK_1.format('dataset_1')
    print(dataset_location)
    train_dataset, test_dataset = load_google_dataset(dataset_location)
    print('load')
    x_train, y_train = get_x_y_as_matrix(train_dataset, nExamples)
    x_test, y_test = get_x_y_as_matrix(test_dataset, nExamples)

    t1 = time.time()
    print('Loaded {} examples in {:.2f}s'.format(x_train.shape, t1 - t0))

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
                    return calculate_activations_by_batches(x_train, x_test, config_path, weights_path,
                                                            nNeurons=nNeurons,
                                                            sample_neurons_strategy=sample_neurons_strategy,
                                                            num_skipped_layers_from_start=1,
                                                            skip_reduction_layers=skip_reduction)

                yield calc_acts, dirname + '_' + str(trained)
