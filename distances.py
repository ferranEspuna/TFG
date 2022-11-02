import functools
from typing import List, Optional, Callable

import numpy as np
from scipy.stats import spearmanr


def soft_step(x, alpha=None):
    # if no argument is passed for alpha, just a regular step function
    if alpha is None:
        return (x > 0 + .5 * (x == 0)).astype(float)
    # if we pass alpha we calculate the smooth version
    return 1 / (1 + np.exp(-alpha * x))


def soft_rank_by_rows(M, alpha=None):
    # numpy broadcasting fuckery to get "soft" matrix of pairwise "greater than" relationships
    greater = soft_step(M[:, None, :] - M[:, :, None], alpha=alpha)

    # sum each matrice's columns to get ranks,
    # subtract diagonal to remove cases where we compare something to itself
    return np.sum(greater, axis=1) - np.diagonal(greater, 0, 1, 2)


# TODO con datos reales hace raíz de negativos! investigar qué pasa
def soft_spearman_distance(M, alpha=None):
    # rank the elements of each row
    ranks = soft_rank_by_rows(M, alpha=alpha)

    # constant involved in spearman's rho
    n = M.shape[1]
    k = 6 / (n * (n ** 2 - 1))

    # more fuckery to get pairwise differences, then calculate spearman's rhos by rows
    rhos = 1 - k * np.sum((ranks[:, None, :] - ranks[None, :, :]) ** 2, axis=2)

    # correlation distance
    d = np.sqrt(1 - rhos ** 2)
    assert d.shape == (M.shape[0], M.shape[0])
    return d


# correlation coefficient-based distance used previously
def pearson_distance(M: np.ndarray) -> np.ndarray:
    d = np.sqrt(1 - np.corrcoef(M) ** 2)
    assert d.shape == (M.shape[0], M.shape[0])
    return d


# regular old euclidean distance in Rn
def euclidean_distance(M: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(M[None, :, :] - M[:, None, :], axis=2) / M.shape[1]
    assert d.shape == (M.shape[0], M.shape[0])
    return d


def spearman_distance(M: np.ndarray) -> np.ndarray:
    return np.sqrt(1 - spearmanr(M.T)[0] ** 2)


def generalized_jaccard_distance(M: np.ndarray) -> np.ndarray:
    intersections = np.sum(np.minimum(M[None, :, :], M[:, None, :]), axis=2)
    unions = np.sum(np.maximum(M[None, :, :], M[:, None, :]), axis=2)

    d = 1 - intersections / unions

    # if they both have 0 measure we set distance to 1, which is the maximum
    np.nan_to_num(d, nan=0.0, copy=False)
    assert d.shape == (M.shape[0], M.shape[0])
    return d


def generalized_jaccard_distance_normalized(M: np.ndarray) -> np.ndarray:
    return generalized_jaccard_distance(M / np.max(M, axis=1)[:, None])


def soft_jaccard_distance_activations(M: np.ndarray, threshold: float = 0, alpha: Optional[float] = None) -> np.ndarray:
    return generalized_jaccard_distance(soft_step(M - threshold, alpha=alpha))


def generalized_jaccard_distance_normalized_activations(M: np.ndarray,
                                                        threshold: float = 0,
                                                        alpha: Optional[float] = None) -> np.ndarray:

    return generalized_jaccard_distance(soft_step(M / np.max(M, axis=1)[:, None] - threshold, alpha=alpha))


def soft_jaccard_distance_ranks(M: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
    return generalized_jaccard_distance(soft_rank_by_rows(M, alpha=alpha))


class Distance:
    def __init__(self, fun: Callable[[np.ndarray], np.ndarray], name: str = ''):
        self.fun: Callable[[np.ndarray], np.ndarray] = fun
        self.name: str = name


def get_all_distances_no_param(alphas: List[Optional[float]], thresholds: List[float]) -> List[Distance]:
    dists_no_param = [Distance(euclidean_distance, 'Eucliedan Distance'),
                      Distance(pearson_distance, 'Pearson Correlation Distance'),
                      Distance(spearman_distance, 'Spearman Correlation distance'),
                      Distance(generalized_jaccard_distance, 'Generalized Jaccard Distance'),
                      Distance(generalized_jaccard_distance_normalized,
                               'Generalized Jaccard Distance for Normaized activations')]

    dists_alpha = [Distance(soft_jaccard_distance_ranks, 'Jaccard Distance on Ranks')]

    dists_alpha_threshold = [Distance(soft_jaccard_distance_activations, 'Jaccard Distance on Activations')]

    for d in dists_alpha:
        for a in alphas:

            name = d.name

            if a is not None:
                name = 'Soft ' + name + ', alpha = ' + str(a)

            d2fun = functools.partial(d.fun, alpha=a)

            dists_no_param.append(Distance(d2fun, name))

    for d in dists_alpha_threshold:
        for t in thresholds:
            for a in alphas:

                name = d.name

                if a is not None:
                    name = 'Soft ' + name + ', alpha = ' + str(a)
                if t is not None:
                    name = name + ', threshold = ' + str(t)

                d2fun = functools.partial(d.fun, alpha=a, threshold=t)
                dists_no_param.append(Distance(d2fun, name))

    return dists_no_param


def get_all_distances_no_param_experiment(alphas: List[Optional[float]], thresholds: List[float]) -> List[Distance]:
    dists_no_param = [Distance(euclidean_distance, 'Eucliedan Distance'),
                      Distance(pearson_distance, 'Pearson Distance'),
                      Distance(generalized_jaccard_distance, 'GJD'),
                      Distance(generalized_jaccard_distance_normalized, 'GJD for Normaized activations')]

    dists_alpha_threshold = [Distance(soft_jaccard_distance_activations, 'GJD on Binary Activations'),
                             Distance(generalized_jaccard_distance_normalized_activations, 'GJD on Binary Normalized Activations')]

    for d in dists_alpha_threshold:
        for t in thresholds:
            for a in alphas:

                name = d.name

                if a is not None:
                    name = 'Soft ' + name + ', alpha = ' + str(a)
                if t is not None:
                    name = name + ', threshold = ' + str(t)

                d2fun = functools.partial(d.fun, alpha=a, threshold=t)
                dists_no_param.append(Distance(d2fun, name))

    return dists_no_param
