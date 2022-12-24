import os.path

from data.data import get_google_examples, get_google_examples_train
from experiment import run_experiments_once
from distances import get_all_distances_no_param_experiment
from summaries import MAX_DIM_NEEDED
from sampling import largest_avg_activation_indices
import time

thresholds = [0.001, 0.5]
nNeurons = 3000
nExamples = 2000
google_example_generator = get_google_examples_train(nExamples, nNeurons,
                                                     sample_neurons_strategy=largest_avg_activation_indices,
                                                     skip_reduction=True)

SAVE_PATH = "./results/Google/final_experiment_fast"

if __name__ == '__main__':

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    for epoch in range(2, 20):

        try:

            savepath_shard = os.path.join(SAVE_PATH, f'epoch_{epoch}')

            for _ in range(96):

                t0 = time.time()
                try:

                    run_experiments_once(activation_generator=google_example_generator, max_dimension=MAX_DIM_NEEDED,
                                         distances=get_all_distances_no_param_experiment(thresholds),
                                         save=True, save_path=savepath_shard)
                    t1 = time.time()
                    print('Computed all in {:.2f}s'.format(t1 - t0))

                except AssertionError:
                    pass
        except:
            pass
