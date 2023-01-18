import os.path

from data.data import get_google_examples_train, TASK_NAMES
from experiment import run_experiments_once
from distances import get_all_distances_no_param_experiment
from summaries import MAX_DIM_NEEDED
from sampling import largest_avg_activation_indices, random_indices
import time

thresholds = [0.5]
nNeurons = 3000
nExamples = 2000

SAVE_PATH = "./results/Google/all_tasks"

if __name__ == '__main__':

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    for task_name in TASK_NAMES:

        try:

            savepath_task = os.path.join(SAVE_PATH, task_name)
            google_example_generator = get_google_examples_train(nExamples, nNeurons, task_name,
                                                                 sample_neurons_strategy=largest_avg_activation_indices,
                                                                 skip_reduction=True)
            for _ in range(200):

                t0 = time.time()
                try:

                    run_experiments_once(activation_generator=google_example_generator, max_dimension=MAX_DIM_NEEDED,
                                         distances=get_all_distances_no_param_experiment(thresholds),
                                         save=True, save_path=savepath_task)
                    t1 = time.time()
                    print('\nComputed model in {:.2f}s\n'.format(t1 - t0))

                except AssertionError:
                    pass
        except Exception as e:
            print(e)
