from data.data import get_google_examples
from experiment import run_experiments_once
from distances import get_all_distances_no_param
from summaries import get_all_summaries, MAX_DIM_NEEDED
from sampling import random_indices, largest_avg_activation_indices
import tracemalloc


alphas = [None]
thresholds = [0]
nNeurons = 500
nExamples = 200
google_example_generator = get_google_examples(nExamples, layer_by_layer=True)

SAVE_PATH = "./results/Google/task1_v5"


if __name__ == '__main__':

    tracemalloc.start()

    for i in range(50):

        activations_model, model_name = google_example_generator.__next__()
        print(model_name)
        run_experiments_once(activations=activations_model, max_dimension=MAX_DIM_NEEDED,
                             distances=get_all_distances_no_param(alphas, thresholds), summaries=get_all_summaries(),
                             samples_neurons=nNeurons, sample_neurons_strategy=largest_avg_activation_indices,
                             vis=True, name=model_name, save=True, save_path=SAVE_PATH)

        mem = tracemalloc.get_traced_memory()
        print("Used memory: {}".format(mem[1] - mem[0]))


