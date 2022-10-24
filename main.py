from data.data import get_google_examples
from experiment import run_experiments_once
from distances import get_all_distances_no_param
from summaries import get_all_summaries, MAX_DIM_NEEDED
from sampling import random_indices, largest_avg_activation_indices

google_example_generator = get_google_examples()

alphas = [None]
thresholds = [0]
nNeurons = 10
nExamples = 10

if __name__ == '__main__':

    for i in range(5):
        activations_first_model, first_model_name = google_example_generator.__next__()

        print(run_experiments_once(activations=activations_first_model, max_dimension=MAX_DIM_NEEDED,
                                   distances=get_all_distances_no_param(alphas, thresholds), summaries=get_all_summaries(),
                                   samples_neurons=nNeurons, samples_examples=nExamples,
                                   sample_neurons_strategy=largest_avg_activation_indices,
                                   vis=True, name=first_model_name))
