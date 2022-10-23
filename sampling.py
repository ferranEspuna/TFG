from typing import Optional, Callable

import numpy as np


# thanks, Ruben
def largest_avg_activation_indices(M: np.ndarray, nSamples: int) -> np.ndarray:
    # find average activation of each neuron
    average_activations_x_neuron = np.average(np.abs(M), axis=1)
    assert average_activations_x_neuron.shape[0] == M.shape[0]

    # select indices of larger avg activation
    selected_indices = np.argpartition(average_activations_x_neuron, -nSamples)[-nSamples:]
    assert len(selected_indices) == nSamples

    # create boolean array
    bool_array = np.array([i in selected_indices for i in range(M.shape[0])])
    assert np.sum(bool_array) == nSamples
    return bool_array


# select nSamples random neurons
def random_indices(M: np.ndarray, nSamples: int) -> np.ndarray:
    bool_array = np.array([True] * nSamples + [False] * (M.shape[0] - nSamples))
    np.random.shuffle(bool_array)
    assert np.sum(bool_array) == nSamples
    return bool_array


def sample_all(activations: np.ndarray, samples_examples: Optional[int] = None, samples_neurons: Optional[int] = None,
               sample_neurons_strategy: Optional[Callable[[np.ndarray, int], np.ndarray]] = random_indices
               ) -> np.ndarray:

    total_neurons, total_examples = activations.shape

    # The indices of the training examples we will look at
    if samples_examples is None or samples_examples > total_examples:
        indices_examples = np.full(total_examples, True)

    else:
        indices_examples = np.array([True] * samples_examples + [False] * (total_examples - samples_examples))
        np.random.shuffle(indices_examples)

    sampled_examples = activations[:, indices_examples]

    # The indices of the neurons we will look at
    if samples_neurons is None or samples_neurons > total_neurons:
        indices_neurons = np.full(total_neurons, True)

    else:
        if sample_neurons_strategy is None:
            raise Exception('Number of neurons passed but no sampling strategy')
        indices_neurons = sample_neurons_strategy(sampled_examples, samples_neurons)

        # This is the data we will perform our persistent homology computations on
    return sampled_examples[indices_neurons, :]
