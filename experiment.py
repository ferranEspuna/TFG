import os.path
from pickle import dump
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from typing import List, Optional, Callable, Tuple
from distances import Distance
from sampling import sample_neurons
from seaborn import displot
from textwrap import wrap

SAVE_PATH_DEFAULT = "./results/Google/task1"


# deterministic setup for an experiment
class ExperimentResult:

    def __init__(self, diagrams: List[np.ndarray], summaries: Tuple[float], name: str = 'ExperimentResult') -> None:
        # self.distance_matrix = distance_matrix
        self.diagrams = diagrams
        self.summaries = summaries
        self.save_dir = None
        self.name = name

    def get_save_dir(self, result_path="./results/Google/task1"):
        if self.save_dir != result_path:
            assert not os.path.isdir(result_path)
            os.mkdir(result_path)
            self.save_dir = result_path

    def save(self, result_path: str = "./results/Google/task1"):

        self.get_save_dir(result_path)
        plot_diagrams(self.diagrams, show=False)
        plt.title('Persistance Diagrams' + self.name, wrap=True)
        plt.savefig(result_path + '/diagrams')
        plt.close()

        for i, diag in enumerate(self.diagrams):

            if i == 0:
                displot([point[1] for point in diag if point[1] != float('inf')], kind='kde')
                plt.title('\n'.join(wrap('H0 death distibution. ' + self.name, 60)))
            else:
                births, deaths = list(zip(*diag))
                displot(x=births, y=deaths, kind='kde', fill=True)
                plt.title('\n'.join(wrap('H{} distribution. '.format(i) + self.name, 60)))

            plt.savefig(result_path + '/distribution_{}'.format(i), bbox_inches="tight")
            plt.close()

            with open(result_path + '/diagram_{}'.format(i), 'wb') as f:
                dump(diag, f)


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

    def run(self, save: Optional[bool] = False,
            save_path: Optional[str] = "./results/Google/task1") -> None:

        # distance matrix of sample
        d = self.dist.fun(self.sample)

        diags = ripser(d, maxdim=self.maxdim, thresh=1, distance_matrix=True)['dgms']
        sums = tuple(summary(diags) for summary in self.summaries)
        self.result = ExperimentResult(name=self.name, diagrams=diags, summaries=sums)

        if save:
            self.result.save(save_path)
            plt.imshow(d)
            plt.title(self.dist.name, wrap=True)
            plt.savefig(save_path + '/distances')
            plt.clf()


def run_experiments_once(activations: np.ndarray, max_dimension: int, distances: List[Distance],
                         summaries: Optional[List[Callable]] = None,
                         samples_neurons: Optional[int] = None,
                         sample_neurons_strategy: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
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
        e.run(save=save, save_path=save_path + '/' + name + '/' + e.dist.name)

    return np.array([e.result.summaries for e in experiments])
