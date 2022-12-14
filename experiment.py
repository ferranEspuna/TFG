import os.path
import time
from pickle import dump
import numpy as np
import matplotlib.pyplot as plt
from persim import plot_diagrams
from typing import List, Optional, Callable, Tuple, Generator
from distances import Distance
from seaborn import displot
from textwrap import wrap
from adapters import ripser_normal

adapter = ripser_normal


# deterministic setup for an experiment
class ExperimentResult:

    def __init__(self, diagrams_train: List[np.ndarray], name: str = 'ExperimentResult') -> None:
        self.diagrams_train = diagrams_train
        self.save_dir = None
        self.name = name

    def get_save_dir(self, result_path="./results/Google/task1"):
        if self.save_dir != result_path:
            assert not os.path.isdir(result_path)
            os.mkdir(result_path)
            self.save_dir = result_path

    @classmethod
    def save_diag(cls, diags, name, result_path, result_string):

        try:
            plot_diagrams(diags, show=False)
            plt.title('Persistance Diagrams for {}. {} dataset'.format(name, result_string), wrap=True)
            plt.savefig(result_path + '/diagrams_{}'.format(result_string))
            plt.close()

        except ValueError:
            print("Could not plot persistance diagrams")

        for i, diag in enumerate(diags):

            try:

                if i == 0:
                    displot([point[1] for point in diag if point[1] != float('inf')], kind='kde')
                    plt.title(
                        '\n'.join(wrap('H0 death distibution for {}. {} dataset'.format(name, result_string), 60)))
                    plt.savefig(result_path + '/distribution_{}_{}'.format(i, result_string), bbox_inches="tight")

                else:

                    births, deaths = list(zip(*diag))
                    displot(x=births, y=deaths, kind='kde', fill=True)
                    plt.title('\n'.join(wrap('H{} distribution for {}. {} dataset'.format(i, name, result_string), 60)))
                    plt.savefig(result_path + '/distribution_{}_{}'.format(i, result_string), bbox_inches="tight")

            except ValueError:
                print("Could not plot H{} distribution".format(i))

            plt.close()

            with open(result_path + '/diagram_{}_{}'.format(i, result_string), 'wb') as f:
                dump(diag, f)

    def save(self, result_path: str = "./results/Google/task1"):
        self.get_save_dir(result_path)
        self.save_diag(self.diagrams_train, self.name, result_path, 'Train')


class Experiment:
    def __init__(self, sample_train: np.ndarray, dist: Distance, name: Optional[str], maxdim: int) -> None:

        self.sample_train: np.ndarray = sample_train
        self.dist: Distance = dist
        self.maxdim: int = maxdim
        self.result: Optional[ExperimentResult] = None

        # naming
        if name is None:
            self.name = dist.name
        else:
            self.name = name

    def run(self, save: Optional[bool] = False,
            save_path: Optional[str] = "./results/Google/task1") -> None:

        t0 = time.time()
        # distance matrix of sample
        d_train = self.dist.fun(self.sample_train)
        t1 = time.time()
        print('computed distances in {:.2f}s'.format(t1 - t0))

        t0 = time.time()
        diags_train = adapter(d_train, self.maxdim)
        t1 = time.time()
        print('computed diagrams in {:.2f}s'.format(t1 - t0))

        self.result = ExperimentResult(name=self.name, diagrams_train=diags_train)

        if save:
            t0 = time.time()

            self.result.save(save_path)

            plt.imshow(d_train)
            plt.title(self.dist.name + '. Train Dataset.', wrap=True)
            plt.savefig(save_path + '/distances_train')
            plt.clf()

            t1 = time.time()
            print('saved data in {:.2f}s'.format(t1 - t0))


def run_experiments_once(
        activation_generator: Generator[Tuple[Callable[[], np.ndarray], str], None, None],
        max_dimension: int, distances: List[Distance],
        save: Optional[bool] = False, save_path: str = ""
) -> None:
    activation_callable, name = activation_generator.__next__()
    if save:

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        general_dir = save_path + '/' + name
        assert not os.path.isdir(general_dir)
        os.mkdir(general_dir)

    print(name)
    sample_matrix_train = activation_callable()

    for dist in distances:

        e = Experiment(sample_matrix_train, dist,
                       name=name + ': ' + dist.name,
                       maxdim=max_dimension)

        print("\n" + e.name)
        try:
            e.run(save=save, save_path=save_path + '/' + name + '/' + e.dist.name)
        except AssertionError:
            pass