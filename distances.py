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


class Distance:
    def __init__(self, fun, name=''):
        self.fun = fun
        self.name = name


def get_all_distances_no_param(alphas, thresholds):
    dists_no_param = [Distance(euclidean_distance, 'Eucliedan Distance'),
                      Distance(pearson_distance, 'Pearson Correlation Distance'),
                      Distance(generalized_jaccard_distance, 'Generalized Jaccard Distance')]

    dists_alpha = [Distance(soft_spearman_distance, 'Spearman Correlation Distance'),
                   Distance(soft_jaccard_distance_ranks, 'Jaccard Distance on Ranks')]

    dists_alpha_threshold = [Distance(soft_jaccard_distance_activations, 'Jaccard Distance on Activations')]

    for a in alphas:

        for d in dists_alpha:

            if a is None:
                name = d.name
            else:
                name = 'Soft ' + d.name + ', alpha = ' + str(a)

            dists_no_param.append(Distance(lambda M: d.fun(M, alpha=a), name=name))

        for t in thresholds:

            for d in dists_alpha_threshold:

                name = d.name

                if a is not None:
                    name = 'Soft ' + name + ', alpha = ' + str(a)
                if t is not None:
                    name = d.name + ', threshold = ' + str(t)

                dists_no_param.append(Distance(lambda M: d.fun(M, alpha=a, threshold=t), name=name))

    return dists_no_param
