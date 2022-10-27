import numpy as np
from data.data import get_google_examples

def test_data_is_same_both_ways():
    lbl = get_google_examples(layer_by_layer=True)
    allatonce = get_google_examples(layer_by_layer=False)

    for i in range(10):
        x, namex = lbl.__next__()
        y, namey = allatonce.__next__()
        print(namex)

        assert x.shape == y.shape
        assert(np.allclose(x, y))