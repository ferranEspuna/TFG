import functools
from typing import List, Optional, Callable
from itertools import chain
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
    np.nan_to_num(d, nan=0.0, copy=False)
    np.fill_diagonal(d, 0)
    d = np.maximum(d, d.T)
    assert d.shape == (M.shape[0], M.shape[0])
    return d


def pearson_distance_noabs(M: np.ndarray) -> np.ndarray:
    d = np.sqrt(1 - np.corrcoef(M))
    np.nan_to_num(d, nan=0.0, copy=False)
    np.fill_diagonal(d, 0)
    d = np.maximum(d, d.T)
    assert d.shape == (M.shape[0], M.shape[0])
    return d


def dist_by_chunks(M, dist, chunck_size=1000):
    n = M.shape[0]
    final_mat = np.zeros((n, n))

    chunk_starts = range(0, n, chunck_size)
    chunk_ends = chain(range(chunck_size, n, chunck_size), [n])

    for start1, end1 in zip(chunk_starts, chunk_ends):

        chunk_starts_2 = range(0, n, chunck_size)
        chunk_ends_2 = chain(range(chunck_size, n, chunck_size), [n])

        for start2, end2 in zip(chunk_starts_2, chunk_ends_2):
            final_mat[start1: end1, start2: end2] = dist(M[start1: end1], M[start2: end2])

    return final_mat


def euclidean_distance_2(M: np.ndarray, N: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(N[None, :, :] - M[:, None, :], axis=2) / M.shape[1]
    assert d.shape == (M.shape[0], N.shape[0])
    return d


# regular old euclidean distance in Rn
def euclidean_distance(M: np.ndarray) -> np.ndarray:
    d = dist_by_chunks(M, euclidean_distance_2)
    assert d.shape == (M.shape[0], M.shape[0])
    return d


def euclidean_distance_normalized(M: np.ndarray) -> np.ndarray:
    return euclidean_distance(M / np.max(M, axis=1)[:, None])


def spearman_distance(M: np.ndarray) -> np.ndarray:
    d = np.sqrt(1 - spearmanr(M.T)[0] ** 2)
    np.nan_to_num(d, nan=0.0, copy=False)
    np.fill_diagonal(d, 0)
    d = np.maximum(d, d.T)
    assert d.shape == (M.shape[0], M.shape[0])
    return d


def spearman_distance_noabs(M: np.ndarray) -> np.ndarray:
    d = np.sqrt(1 - spearmanr(M.T)[0])
    np.nan_to_num(d, nan=0.0, copy=False)
    np.fill_diagonal(d, 0)
    d = np.maximum(d, d.T)
    assert d.shape == (M.shape[0], M.shape[0])
    return d


def generalized_jaccard_distance_2(M: np.ndarray, N: np.ndarray) -> np.ndarray:
    intersections = np.sum(np.minimum(N[None, :, :], M[:, None, :]), axis=2)
    unions = np.sum(np.maximum(N[None, :, :], M[:, None, :]), axis=2)
    d = 1 - intersections / unions
    # if they both have 0 measure we set distance to 1, which is the maximum
    np.nan_to_num(d, nan=0.0, copy=False)
    assert d.shape == (M.shape[0], N.shape[0])
    return d


def generalized_jaccard_distance(M: np.ndarray) -> np.ndarray:
    d = dist_by_chunks(M, generalized_jaccard_distance_2)
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


def fast_jaccard_distance(M: np.ndarray, threshold: Optional[float] = 0) -> np.ndarray:
    M_yes = (M > threshold).astype(float)
    M_no = 1 - M_yes
    intersection = M_yes @ M_yes.T
    union = M.shape[1] - M_no @ M_no.T

    return 1 - intersection / union


def intersection_distance(M: np.ndarray, threshold: Optional[float] = 0) -> np.ndarray:
    M_yes = (M > threshold).astype(float)
    intersection = M_yes @ M_yes.T
    return 1 - intersection / M.shape[1]


def fast_jaccard_distance_normalized_activations(M: np.ndarray, threshold: float = 0) -> np.ndarray:
    return fast_jaccard_distance(M / np.max(M, axis=1)[:, None], threshold)


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


def get_all_distances_no_param_experiment(thresholds: List[float]) -> List[Distance]:
    dists_no_param = [Distance(pearson_distance, 'Pearson Distance'),
                      Distance(spearman_distance, 'Spearman Distance')]

    dists_threshold = [
        Distance(fast_jaccard_distance_normalized_activations, 'Jaccard Distance on Binary Normalized Activations'),
        Distance(intersection_distance, 'Intersection Distance on Binary Normalized Activations')]

    for d in dists_threshold:
        for t in thresholds:
            name = d.name
            name = name + ', threshold = ' + str(t)

            d2fun = functools.partial(d.fun, threshold=t)
            dists_no_param.append(Distance(d2fun, name))

    return dists_no_param


def get_all_distances_no_param_no_abs_experiment() -> List[Distance]:
    dists_no_param = [Distance(pearson_distance_noabs, 'Pearson Distance no abs'),
                      Distance(spearman_distance_noabs, 'Spearman Distance no abs'), ]

    return dists_no_param


def get_all_gjd_no_param_experiment(alphas: List[Optional[float]], thresholds: List[float]) -> List[Distance]:
    dists_no_param = []

    dists_alpha_threshold = [
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
