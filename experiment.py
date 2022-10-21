import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams


def run_experiment_once(activations, max_dimension, distances, summaries,  # Obligatory params
                        samples_neurons=None, samples_examples=None, sample_neurons_strategy=None,  # Sampling
                        vis=False, verbose=False):  # User interaction

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
        indices_neurons = sample_neurons_strategy(sampled_examples, samples_neurons)

    # This is the data we will perform our persistent homology computations on
    sample_matrix = sampled_examples[indices_neurons, :]

    if verbose:
        print("Shape of sample matrix: ", sample_matrix.shape)

    # Calculate distance matrices according to our input metrics
    distance_matrices = [dist(sample_matrix) for dist in distances]
    if vis:
        for mat in distance_matrices:
            plt.imshow(mat)
            plt.show()

    # Calculate the corresponding persistence diagrams using ripser
    diagrams = [ripser(mat, max_dimension, thresh=1, distance_matrix=True)['dgms'] for mat in distance_matrices]

    # Visualize the diagrams if specified
    if vis:
        for diag in diagrams:
            plot_diagrams(diag)
            plt.show()

    # Calculate each summary for each distance and return them as a numpy 2D array
    return np.array([[summary(diag) for summary in summaries] for diag in diagrams])
