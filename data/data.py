import os
from typing import Generator

import numpy as np
import tensorflow_datasets as tfds
from data.Google.GoogleDatasetReader import load_google_dataset
from data.Google.GoogleModelReader import load_google_model

FOLDER_TEMPLATE_TASK_1 = "./data/Google/public_data/input_data/task1_v4/{}"
MAX_EXAMPLES = 200


def get_data_test() -> np.ndarray:
    num_neurons = 200
    num_examples = 200
    M = np.random.random((num_neurons, num_examples))
    return M


def load_smol_input(x, model, num_skipped_layers_from_start=1):
    activations = []
    skipped_iterations = 0
    for layer in model.layers:
        x = layer(x)

        if skipped_iterations < num_skipped_layers_from_start:
            skipped_iterations += 1
        else:
            examples_x_neurons = np.reshape(np.copy(x.numpy()), newshape=(-1, x.shape[0]))
            activations.append(examples_x_neurons)

    final_array = np.concatenate(activations, axis=0)
    return final_array


def get_google_examples() -> Generator[np.ndarray, None, None]:
    dataset_location = FOLDER_TEMPLATE_TASK_1.format('dataset_1')

    train_dataset, test_dataset = load_google_dataset(dataset_location)
    npiterator = tfds.as_numpy(train_dataset.take(MAX_EXAMPLES))
    x_train_list, y_train_list = zip(*npiterator)

    x_train_list = list(map(lambda x_example: x_example[np.newaxis, ...], x_train_list))
    x_train = np.concatenate(x_train_list, axis=0)

    for i in range(800):
        dirname = 'model_' + str(i)
        model_location = FOLDER_TEMPLATE_TASK_1.format(dirname)
        if os.path.isdir(model_location):
            model = load_google_model(model_location)
            yield load_smol_input(x_train, model, num_skipped_layers_from_start=1), dirname
