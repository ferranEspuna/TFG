import logging
import tracemalloc

from data.data import get_google_examples
from experiment import run_experiments_once
from distances import get_all_distances_no_param_experiment
from summaries import get_all_summaries, MAX_DIM_NEEDED
from sampling import random_indices, largest_avg_activation_indices
import time

alphas = [None, 10]
thresholds = [0.001, 0.5, 0.8]
nNeurons = 3000
nExamples = 2000
google_example_generator = get_google_examples(nExamples, layer_by_layer=True, skip_reduction=True)

SAVE_PATH = "./results/Google/task1"

if __name__ == '__main__':
    tracemalloc.start()

    for i in range(200):

        t0 = time.time()
        try:
            run_experiments_once(activation_generator=google_example_generator, max_dimension=MAX_DIM_NEEDED,
                                 distances=get_all_distances_no_param_experiment(alphas, thresholds),
                                 summaries=get_all_summaries(),
                                 samples_neurons=nNeurons, sample_neurons_strategy=largest_avg_activation_indices,
                                 save=True, save_path=SAVE_PATH)
            t1 = time.time()
            print(t1 - t0)

        except AssertionError:
            pass
