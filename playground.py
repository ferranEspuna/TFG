import numpy as np
from data import get_data
from distances import soft_spearman_distance, pearson_distance
from matplotlib import pyplot as plt

data = get_data()
D1 = pearson_distance(data ** 9)
D2 = soft_spearman_distance(data ** 9, alpha=None)

plt.imshow(D1)
plt.show()
plt.imshow(D2)
plt.show()

print(np.mean(np.abs(D2-D1)))
