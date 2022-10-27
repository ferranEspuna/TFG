import os.path

import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from typing import List, Optional, Callable, Tuple
from distances import Distance
from sampling import sample_neurons

SAVE_PATH_DEFAULT = "./results/Google/task1"


# deterministic setup for an experiment
class ExperimentResult:

    def __init__(self, distance_matrix: np.ndarray, diagrams: List[np.ndarray], summaries: Tuple[float]) -> None:
        #self.distance_matrix = distance_matrix
        self.diagrams = diagrams
        self.summaries = summaries
        self.save_dir = None

    def get_save_dir(self, result_path="./results/Google/task1"):
        if self.save_dir != result_path:
            assert not os.path.isdir(result_path)
            os.mkdir(result_path)
            self.save_dir = result_path

    def save(self, result_path: str = "./results/Google/task1"):
        self.get_save_dir(result_path)
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

    def run(self, vis: Optional[bool] = False, save: Optional[bool] = False, save_path: Optional[str] = "./results/Google/task1") -> None:

        # distance matrix of sample
        D = self.dist.fun(self.sample)
        if vis:
            plt.imshow(D)
            plt.title(self.dist.name)
            plt.show()

        diags = ripser(D, maxdim=self.maxdim, thresh=1, distance_matrix=True)['dgms']
        sums = tuple(summary(diags) for summary in self.summaries)
        self.result = ExperimentResult(distance_matrix=D, diagrams=diags, summaries=sums)

        if vis or save:
            plot_diagrams(diags, show=vis)
            plt.title(self.name)

        if save:

            self.result.get_save_dir(save_path)
            plt.savefig(save_path + '/diagrams')

        plt.clf()


def run_experiments_once(activations: np.ndarray, max_dimension: int, distances: List[Distance],
                         summaries: Optional[List[Callable]] = None,
                         samples_neurons: Optional[int] = None, samples_examples: Optional[int] = None,
                         sample_neurons_strategy: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
                         vis: Optional[bool] = False,
                         name: Optional[str] = '', save: Optional[bool] = False, save_path: str = SAVE_PATH_DEFAULT
                         ) -> np.ndarray:

    if save and not os.path.isdir(save_path):
        os.mkdir(save_path)

    sample_matrix = sample_neurons(activations, samples_neurons,
                               strategy=sample_neurons_strategy)

    experiments = [Experiment(sample_matrix, dist,
                              name=name + ': ' + dist.name,
                              maxdim=max_dimension, summaries=summaries
                              )
                   for dist in distances]

    if save:
        general_dir = save_path + '/' + name
        assert not os.path.isdir(general_dir)
        os.mkdir(general_dir)

    for e in experiments:
        e.run(vis=vis, save=save, save_path=save_path + '/' + name + '/' + e.dist.name)

    return np.array([e.result.summaries for e in experiments])
