import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from typing import List, Optional, Callable
from distances import Distance


def run_experiment_once(activations: np.ndarray, max_dimension: int, distances: List[Distance],
                        summaries: Optional[List[Callable]] = None,
                        samples_neurons: Optional[int] = None, samples_examples: Optional[int] = None,
                        sample_neurons_strategy: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,  # Sampling
                        vis: Optional[bool] = False, verbose: Optional[bool] = False):  # User interaction

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
    sample_matrix = sampled_examples[indices_neurons, :]

    if verbose:
        print("Shape of sample matrix: ", sample_matrix.shape)

    # Calculate distance matrices according to our input metrics
    distance_matrices = [dist.fun(sample_matrix) for dist in distances]
    distance_names = [dist.name for dist in distances]

    if vis:
        for mat, name in zip(distance_matrices, distance_names):
            plt.imshow(mat)
            plt.title(name)
            plt.show()

    # Calculate the corresponding persistence diagrams using ripser
    diagrams = [ripser(mat, max_dimension, thresh=1, distance_matrix=True)['dgms'] for mat in distance_matrices]

    # Visualize the diagrams if specified
    if vis:
        for diag, name in zip(diagrams, distance_names):
            plot_diagrams(diag)
            plt.title(name)
            plt.show()

    # Calculate each summary for each distance and return them as a numpy 2D array
    return np.array([[summary(diag) for summary in summaries] for diag in diagrams])
