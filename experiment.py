import os.path
from pickle import dump
import numpy as np
import matplotlib.pyplot as plt
from persim import plot_diagrams
from typing import List, Optional, Callable, Tuple, Generator
from distances import Distance
from sampling import sample_neurons
from seaborn import displot
from textwrap import wrap
from adapters import ripser_plusplus, ripser_normal

adapter = ripser_plusplus
SAVE_PATH_DEFAULT = "./results/Google/task1"


# deterministic setup for an experiment
class ExperimentResult:

    def __init__(self, diagrams_train: List[np.ndarray], diagrams_test: List[np.ndarray],
                 name: str = 'ExperimentResult') -> None:
        self.diagrams_train = diagrams_train
        self.diagrams_test = diagrams_test
        self.save_dir = None
        self.name = name

    def get_save_dir(self, result_path="./results/Google/task1"):
        if self.save_dir != result_path:
            assert not os.path.isdir(result_path)
            os.mkdir(result_path)
            self.save_dir = result_path

    @classmethod
    def save_diag(cls, diags, name, result_path, result_string):
        plot_diagrams(diags, show=False)
        plt.title('Persistance Diagrams for {}. {} dataset'.format(name, result_string), wrap=True)
        plt.savefig(result_path + '/diagrams_{}'.format(result_string))
        plt.close()

        for i, diag in enumerate(diags):

            if i == 0:
                displot([point[1] for point in diag if point[1] != float('inf')], kind='kde')
                plt.title('\n'.join(wrap('H0 death distibution for {}. {} dataset'.format(name, result_string), 60)))
                plt.savefig(result_path + '/distribution_{}_{}'.format(i, result_string), bbox_inches="tight")

            else:
                try:
                    births, deaths = list(zip(*diag))
                    displot(x=births, y=deaths, kind='kde', fill=True)
                    plt.title('\n'.join(wrap('H{} distribution for {}. {} dataset'.format(i, name, result_string), 60)))
                    plt.savefig(result_path + '/distribution_{}_{}'.format(i, result_string), bbox_inches="tight")
                except ValueError:
                    pass

            plt.close()

            with open(result_path + '/diagram_{}_{}'.format(i, result_string), 'wb') as f:
                dump(diag, f)

    def save(self, result_path: str = "./results/Google/task1"):
        self.get_save_dir(result_path)
        self.save_diag(self.diagrams_train, self.name, result_path, 'Train')
        self.save_diag(self.diagrams_test, self.name, result_path, 'Test')


class Experiment:
    def __init__(self, sample_train: np.ndarray, sample_test: np.ndarray, dist: Distance, name: Optional[str],
                 maxdim: int,
                 summaries: Optional[Tuple[Callable[[List[np.ndarray]], float]]] = ()) -> None:

        self.sample_train: np.ndarray = sample_train
        self.sample_test = sample_test
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
        d_train = self.dist.fun(self.sample_train)
        d_test = self.dist.fun(self.sample_test)

        diags_train = adapter(d_train, self.maxdim)
        diags_test = adapter(d_test, self.maxdim)
        self.result = ExperimentResult(name=self.name, diagrams_train=diags_train, diagrams_test=diags_test)

        if save:
            self.result.save(save_path)

            plt.imshow(d_train)
            plt.title(self.dist.name + '. Train Dataset.', wrap=True)
            plt.savefig(save_path + '/distances_train')
            plt.clf()

            plt.imshow(d_test)
            plt.title(self.dist.name + '. Test Dataset', wrap=True)
            plt.savefig(save_path + '/distances_test')
            plt.clf()


def run_experiments_once(
        activation_generator: Generator[Tuple[Callable[[], np.ndarray], Callable[[], np.ndarray], str], None, None],
        max_dimension: int, distances: List[Distance],
        summaries: Optional[List[Callable]] = None,
        samples_neurons: Optional[int] = None,
        sample_neurons_strategy: Optional[Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = None,
        save: Optional[bool] = False, save_path: str = SAVE_PATH_DEFAULT
        ) -> None:

    activation_callable_train, activation_callable_test, name = activation_generator.__next__()

    if save:

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        general_dir = save_path + '/' + name
        assert not os.path.isdir(general_dir)
        os.mkdir(general_dir)

    print(name)
    sample_matrix_train, sample_matrix_test = sample_neurons(activation_callable_train(), activation_callable_test(),
                                                             samples_neurons, strategy=sample_neurons_strategy)

    experiments = [Experiment(sample_matrix_train, sample_matrix_test, dist,
                              name=name + ': ' + dist.name,
                              maxdim=max_dimension, summaries=summaries
                              )
                   for dist in distances]

    for e in experiments:

        try:
            e.run(save=save, save_path=save_path + '/' + name + '/' + e.dist.name)
        except AssertionError:
            pass
