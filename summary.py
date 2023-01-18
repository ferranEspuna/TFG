import glob
import json
import os
import pickle
import statistics
from data.data import TASK_NAMES

from distances import get_all_distances_no_param_experiment, get_all_distances_no_param_no_abs_experiment, \
    get_all_gjd_no_param_experiment
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, kurtosis, kendalltau, pearsonr
from statsmodels.stats.weightstats import DescrStatsW

SAVE_PATH_TEMPLATE = 'results/Google/{}/{}/{}_True/{}/diagram_0_Train'
IMAGE_PATH_TEMPLATE = 'results/Google/{}/{}/{}_True/{}/distribution_0_Train.png'
SUMMARY_PATH_TEMPLATE = 'results/Google/{}/summaries/{}/{}.predict'
DISTANCE_FOLDER_TEMPLATE = 'results/Google/{}/summaries/{}'
MODEL_CONFIGS_PATH_TEMPLATE = 'Google/all_data/reference_data/{}/model_configs.json'
DATA_PATH_TEMPLATE = 'Google/all_data/input_data/{}/model_*'

experiment = 'all_tasks'
thresholds = [0.5]


def summary(diags):
    sum_lifespan = 0
    sum_coso = 0
    sum_dev = 0
    sum_dev_weighted = 0

    for diag in diags:
        births, deaths = zip(*[x for x in diag if x[1] != float('inf')])
        midlives = [(d + b) / 2 for b, d in zip(births, deaths)]
        lifespans = [d - b for b, d in zip(births, deaths)]

        #sum_lifespan += sum(deaths) - sum(births)
        #sum_coso += statistics.stdev(m * l for m, l in zip(midlives, lifespans))
        #sum_dev_weighted += DescrStatsW(births, weights=lifespans, ddof=1).quantile([.5], return_pandas=False)[0]
        sum_dev += statistics.stdev(deaths)

    return [sum_dev / len(diags)]


distance_names = [d.name for d in get_all_distances_no_param_experiment(thresholds)]

for task_name in TASK_NAMES:

    print(f'\n{task_name}')

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
            path_to_result = SAVE_PATH_TEMPLATE.format(experiment, task_name, model_name, dname)
            try:
                with open(os.path.abspath(path_to_result), 'rb') as f:
                    diags.append(pickle.load(f))

            except Exception as e:
                print(e)
            else:
                summ = summary(diags)
                summaries[i].append(summ)

    for i, dname in enumerate(distance_names):

        print(f'{dname}: {spearmanr(gen_gaps, summaries[i])}')


        #para guardar en el formato que dice google. Crear la carpeta 'summaries' donde se hayan guardado los diagramas antes
        """ 
        sum_dict = dict(zip(model_names, summaries[i]))
        
        dname_short = dname.split()[0]

        try:
            os.mkdir(DISTANCE_FOLDER_TEMPLATE.format(experiment, dname_short))
        except:
            pass

       
        fname = SUMMARY_PATH_TEMPLATE.format(experiment, dname_short, task_name)

        with open(fname, 'w') as f:
            f.write(json.dumps(sum_dict))
            
        print(f'summaries stored at {fname}')
            
        """