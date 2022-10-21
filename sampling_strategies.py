import numpy as np


# thanks, Ruben
def largest_avg_activation_indices(M, nSamples):
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
def random_indices(M, nSamples):
    bool_array = np.array([True] * nSamples + [False] * (M.shape[0] - nSamples))
    np.random.shuffle(bool_array)
    assert np.sum(bool_array) == nSamples
    return bool_array
