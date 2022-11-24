from typing import Optional, Callable, Tuple

import numpy as np


# thanks, Ruben
def largest_avg_activation_indices(M1: np.ndarray, M2: np.ndarray, nSamples: int) -> np.ndarray:

    total_neurons = M1.shape[0]
    if nSamples is None or nSamples > total_neurons:
        return np.full(total_neurons, True)

    # find average activation of each neuron
    average_activations_x_neuron_1 = np.average(np.abs(M1), axis=1)
    average_activations_x_neuron_1 /= average_activations_x_neuron_1.mean()
    average_activations_x_neuron_2 = + np.average(np.abs(M2), axis=1)
    average_activations_x_neuron_2 /= average_activations_x_neuron_1.mean()

    average_activations_x_neuron = average_activations_x_neuron_1 + average_activations_x_neuron_2

    assert average_activations_x_neuron.shape[0] == total_neurons

    # select indices of larger avg activation
    selected_indices = np.argpartition(average_activations_x_neuron, -nSamples)[-nSamples:]
    assert len(selected_indices) == nSamples

    # create boolean array
    bool_array = np.array([i in selected_indices for i in range(total_neurons)])
    assert np.sum(bool_array) == nSamples
    return bool_array


# select nSamples random neurons
def random_indices(M1: np.ndarray, M2: np.ndarray, nSamples: int) -> np.ndarray:

    total_neurons = M1.shape[0]
    if nSamples is None or nSamples > total_neurons:
        return np.full(total_neurons, True)

    bool_array = np.array([True] * nSamples + [False] * (M1.shape[0] - nSamples))
    np.random.shuffle(bool_array)
    assert np.sum(bool_array) == nSamples
    return bool_array
