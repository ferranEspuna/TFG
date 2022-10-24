import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from typing import List, Optional, Callable, Tuple
from distances import Distance
from sampling import sample_all


# deterministic setup for an experiment
class ExperimentResult:

    def __init__(self, distance_matrix: np.ndarray, diagrams: List[np.ndarray], summaries: Tuple[float]) -> None:
        self.distance_matrix = distance_matrix
        self.diagrams = diagrams
        self.summaries = summaries

    # TODO
    def save(self, result_path):
        pass


class Experiment:
    def __init__(self, sample: np.ndarray, dist: Distance, name: Optional[str], maxdim: int,
                 summaries: Optional[Tuple[Callable[[List[np.ndarray]], float]]] = ()) -> None:

        self.sample: np.ndarray = sample
        self.dist: Distance = dist
        self.summaries = summaries
        self.maxdim: int = maxdim
        self.result: Optional[ExperimentResult] = None

        # naming
        if name is None:
            self.name = dist.name
        else:
            self.name = name

    def run(self, vis: Optional[bool] = False) -> None:

        # distance matrix of sample
        D = self.dist.fun(self.sample)
        if vis:
            plt.imshow(D)
            plt.title(self.dist.name)
            plt.show()

        diags = ripser(D, maxdim=self.maxdim, thresh=1, distance_matrix=True)['dgms']
        if vis:
            plot_diagrams(diags)
            plt.title(self.name)
            plt.show()

        sums = tuple(summary(diags) for summary in self.summaries)

        self.result = ExperimentResult(distance_matrix=D, diagrams=diags, summaries=sums)


def run_experiments_once(activations: np.ndarray, max_dimension: int, distances: List[Distance],
                         summaries: Optional[List[Callable]] = None,
                         samples_neurons: Optional[int] = None, samples_examples: Optional[int] = None,
                         sample_neurons_strategy: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
                         vis: Optional[bool] = False,
                         name: Optional[str] = '', save: Optional[bool] = False, save_path: Optional[str] = '../results'
                         ) -> np.ndarray:

    sample_matrix = sample_all(activations, samples_examples, samples_neurons,
                               sample_neurons_strategy=sample_neurons_strategy)

    experiments = [Experiment(sample_matrix, dist,
                              name=name + ': ' + dist.name,
                              maxdim=max_dimension, summaries=summaries
                              )
                   for dist in distances]

    for r in experiments:
        r.run(vis=vis)
        if save:
            r.result.save(save_path + '/' + name + '/' + r.dist.name)

    return np.array([e.result.summaries for e in experiments])
