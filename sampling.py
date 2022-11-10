from typing import Optional, Callable, Tuple

import numpy as np


# thanks, Ruben
def largest_avg_activation_indices(M1: np.ndarray, M2: np.ndarray, nSamples: int) -> np.ndarray:
    # find average activation of each neuron
    average_activations_x_neuron_1 = np.average(np.abs(M1), axis=1)
    average_activations_x_neuron_1 /= average_activations_x_neuron_1.mean()
    average_activations_x_neuron_2 = + np.average(np.abs(M2), axis=1)
    average_activations_x_neuron_2 /= average_activations_x_neuron_1.mean()

    average_activations_x_neuron = average_activations_x_neuron_1 + average_activations_x_neuron_2

    assert average_activations_x_neuron.shape[0] == M1.shape[0]

    # select indices of larger avg activation
    selected_indices = np.argpartition(average_activations_x_neuron, -nSamples)[-nSamples:]
    assert len(selected_indices) == nSamples

    # create boolean array
    bool_array = np.array([i in selected_indices for i in range(M1.shape[0])])
    assert np.sum(bool_array) == nSamples
    return bool_array


# select nSamples random neurons
def random_indices(M1: np.ndarray, M2: np.ndarray, nSamples: int) -> np.ndarray:
    bool_array = np.array([True] * nSamples + [False] * (M1.shape[0] - nSamples))
    np.random.shuffle(bool_array)
    assert np.sum(bool_array) == nSamples
    return bool_array


def sample_neurons(activations_train: np.ndarray, activations_test: np.ndarray, samples_neurons: Optional[int] = None,
                   strategy: Optional[Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = random_indices
                   ) -> Tuple[np.ndarray, np.ndarray]:

    total_neurons, total_examples = activations_train.shape

    # The indices of the neurons we will look at
    if samples_neurons is None or samples_neurons > total_neurons:
        indices_neurons = np.full(total_neurons, True)

    else:
        if strategy is None:
            raise Exception('Number of neurons passed but no sampling strategy')
        indices_neurons = strategy(activations_train, activations_test, samples_neurons)

        # This is the data we will perform our persistent homology computations on
    return activations_train[indices_neurons, :], activations_test[indices_neurons, :]
