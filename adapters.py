import numpy as np
import pandas as pd
from ripser import ripser

"""
def ripser_plusplus(M, upper_dim):
    dgs = rpp_py.run("--format distance --dim {} ".format(upper_dim), M)
    dgs_formatted = [pd.DataFrame(dgs[i]).to_numpy() for i in range(upper_dim + 1)]
    return dgs_formatted
"""


def ripser_normal(M, upper_dim):
    return ripser(M, maxdim=upper_dim, thresh=1, distance_matrix=True)['dgms']
