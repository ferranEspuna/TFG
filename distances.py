import numpy as np


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


def soft_spearman_distance(M, alpha=None):
    # rank the elements of each row
    ranks = soft_rank_by_rows(M, alpha=alpha)

    # constant involved in spearman's rho
    n = M.shape[1]
    k = 6 / (n * (n ** 2 - 1))

    # more fuckery to get pairwise differences, then calculate spearman's rhos by rows
    rhos = 1 - k * np.sum((ranks[:, None, :] - ranks[None, :, :]) ** 2, axis=2)

    # correlation distance
    D = np.sqrt(1 - rhos ** 2)
    assert D.shape == (M.shape[0], M.shape[0])
    return D


# correlation coefficient-based distance used previously
def pearson_distance(M):
    D = np.sqrt(1 - np.corrcoef(M) ** 2)
    assert D.shape == (M.shape[0], M.shape[0])
    return D


# regular old euclidean distance in Rn
def euclidean_distance(M):
    D = np.linalg.norm(M[None, :, :] - M[:, None, :], axis=2) / M.shape[1]
    assert D.shape == (M.shape[0], M.shape[0])
    return D


def generalized_jaccard_distance(M):
    intersections = np.sum(np.minimum(M[None, :, :], M[:, None, :]), axis=2)
    unions = np.sum(np.maximum(M[None, :, :], M[:, None, :]), axis=2)

    D = 1 - intersections / unions

    # if they both have 0 measure we set distance to 1, which is the maximum
    np.nan_to_num(D, nan=0.0, copy=False)
    assert D.shape == (M.shape[0], M.shape[0])
    return D


def soft_jaccard_distance_activations(M, threshold=0, alpha=None):
    return generalized_jaccard_distance(soft_step(M - threshold, alpha=alpha))


def soft_jaccard_distance_ranks(M, alpha=None):
    return generalized_jaccard_distance(soft_rank_by_rows(M, alpha=alpha))


def get_all_distances_no_param(alphas, thresholds):

    dists_no_param = [euclidean_distance, pearson_distance, generalized_jaccard_distance]

    dists_alpha = [soft_spearman_distance, soft_jaccard_distance_ranks]

    dists_alpha_threshold = [soft_jaccard_distance_activations]

    for a in alphas:

        for d in dists_alpha:
            dists_no_param.append(lambda M: d(M, alpha=a))

        for t in thresholds:

            for d in dists_alpha_threshold:
                dists_no_param.append(lambda M: d(M, alpha=a, threshold=t))

    return dists_no_param
