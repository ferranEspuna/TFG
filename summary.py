import glob
import json
import os
import pickle
import statistics
from math import sqrt, log
import diptest
import numpy as np
from scipy.stats import shapiro
from data.data import TASK_NAMES

from main import thresholds
from distances import get_all_distances_no_param_experiment, get_all_distances_no_param_no_abs_experiment, \
    get_all_gjd_no_param_experiment
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, kurtosis, kendalltau

DATA_PATH_TEMPLATE = 'Google/all_data/input_data/{}/model_*'
SAVE_PATH_TEMPLATE = 'results/Google/all_tasks/{}/{}_True/{}/diagram_0_Train'
SUMMARY_PATH_TEMPLATE = 'results/Google/all_tasks/summaries/{}/{}.predict'
DISTANCE_FOLDER_TEMPLATE = 'results/Google/all_tasks/summaries/{}'
MODEL_CONFIGS_PATH_TEMPLATE = 'Google/all_data/reference_data/{}/model_configs.json'


def summary(diags):
    # sum_kurt = 0
    sum_dev = 0
    # sum_mean = 0
    # sum_dev_2 = 0
    # sum_mean_2 = 0
    # sum_shapiro = 0
    diags = [diags[0]]
    for diag in diags:
        deaths = [d for b, d in diag if d != float('inf')]
        deaths_2 = [d for b, d in diag if d != float('inf')]

        # deaths_2 = [d ** 2 for d in deaths]
        # sum_kurt += kurtosis(deaths)
        sum_dev += statistics.stdev(deaths)
        # sum_mean += statistics.mean(deaths)
        # sum_dev_2 += statistics.stdev(deaths_2)
        # sum_mean_2 += statistics.mean(deaths_2)
        # sum_shapiro += shapiro(deaths)[0]

    return sum_dev / len(diags)

distance_names = [d.name for d in get_all_distances_no_param_experiment(thresholds)]

for task_name in TASK_NAMES:

    data_path = DATA_PATH_TEMPLATE.format(task_name)
    model_names = [path.split('\\')[-1] for path in glob.glob(data_path)]

    summaries = [[] for _ in distance_names]

    with open(MODEL_CONFIGS_PATH_TEMPLATE.format(task_name)) as f:
        model_config = json.load(f)

    gen_gaps = []

    for model_name in model_names:

        model_idx = model_name.split('_')[-1]
        metrics = model_config[model_idx]['metrics']
        gen_gap = metrics['train_acc'] - metrics['test_acc']
        gen_gaps.append(gen_gap)

        for i, dname in enumerate(distance_names):
            diags = []
            path_to_result = SAVE_PATH_TEMPLATE.format(task_name, model_name, dname)
            try:
                with open(os.path.abspath(path_to_result), 'rb') as f:
                    diags.append(pickle.load(f))
                summ = summary(diags)
                summaries[i].append(summ)
            except:
                pass

    for i, dname in enumerate(distance_names):
        sum_dict = dict(zip(model_names, summaries[i]))

        dname_short = dname.split()[0]

        try:
            os.mkdir(DISTANCE_FOLDER_TEMPLATE.format(dname_short))
        except:
            pass

        with open(SUMMARY_PATH_TEMPLATE.format(dname_short, task_name), 'w') as f:
            f.write(json.dumps(sum_dict))

        print(f'{dname}, {task_name}: {spearmanr(gen_gaps, summaries[i])}')

        """
    
        #print(f'{dname}: {spearmanr(gen_gaps, summaries[i])}')
        #plt.scatter(summaries[i], gen_gaps)
        #plt.show()
    
        reg = LinearRegression()
        reg.fit(summaries[i][:70], [[x] for x in gen_gaps[:70]])
        print(reg.intercept_)
        print(reg.coef_)
        gaps_pred = reg.predict(summaries[i][70:])
    
        plt.scatter(gen_gaps[70:], gaps_pred)
        plt.title(dname)
        plt.show()
        print(f'{dname}: {r2_score(gen_gaps[70:], gaps_pred)}')
        """
