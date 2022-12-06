import glob
import json
import os
import pickle
import statistics
from main import alphas, thresholds
from distances import get_all_distances_no_param_experiment, get_all_distances_no_param_no_abs_experiment
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

DATA_PATH = 'Google/public_data/input_data/task1_v4'
SAVE_PATH = 'results/Google/task1_dim_0_noabs'
MODEL_CONFIGS_PATH = 'Google/public_data/reference_data/task1_v4/model_configs.json'


def summary(diag):
    deaths = [d for b, d in diag if d != float('inf')]
    return statistics.stdev(deaths), statistics.mean(deaths)


model_names = [path.split('/')[-1] for path in glob.glob(DATA_PATH + '/model_*')]

distance_names = [d.name for d in get_all_distances_no_param_no_abs_experiment()]
summaries = [[] for _ in distance_names]

with open(MODEL_CONFIGS_PATH) as f:
    model_config = json.load(f)

gen_gaps = []

for model_name in model_names:

    model_idx = model_name.split('_')[-1]
    metrics = model_config[model_idx]['metrics']
    gen_gap = metrics['train_acc'] - metrics['test_acc']
    gen_gaps.append(gen_gap)

    for i, dname in enumerate(distance_names):
        path_to_result = SAVE_PATH + '/' + model_name + '_True/' + dname + '/diagram_0_Train'

        with open(os.path.abspath(path_to_result), 'rb') as f:
            diag = pickle.load(f)
            summ = summary(diag)
            summaries[i].append(summ)


for i, dname in enumerate(distance_names):
    reg = LinearRegression().fit(summaries[i], [[x] for x in gen_gaps])
    gaps_pred = reg.predict(summaries[i])
    print(f'{dname}: {r2_score(gen_gaps, gaps_pred)}')
