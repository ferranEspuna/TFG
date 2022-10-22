import data
from experiment import run_experiment_once
from distances import get_all_distances_no_param
from summaries import get_all_summaries, MAX_DIM_NEEDED
from sampling_strategies import random_indices

alphas = [None, 0.1, 1, 10, 100]
thresholds = [0.1, 0.5, 0.9]
nNeurons = 20
nExamples = 50

if __name__ == '__main__':
    print(run_experiment_once(activations=data.get_data(), max_dimension=MAX_DIM_NEEDED,
                              distances=get_all_distances_no_param(alphas, thresholds), summaries=get_all_summaries(),
                              samples_neurons=nNeurons, samples_examples=nExamples,
                              sample_neurons_strategy=random_indices,
                              vis=True, verbose=False))
