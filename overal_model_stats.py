import numpy as np
import pandas as pd
import argparse
import utils
import os

def remove_dup_runs(df):
    # repeated experiments are annoying. Use to keep only the most recent records in the file
    # get index for duplicate generations, remove these later occurances, keeping the first ones
    df = df.drop_duplicates(keep='last', subset=['generation'], ignore_index=True)
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ColoredMNIST', help='Dataset')
parser.add_argument('--model', type=str, default='ano_label', help='Name of model to analyse')
parser.add_argument('--res-dir', type=str, default='./results', help='Overall directory where result csvs are stored.')
parser.add_argument('--is-valid', type=int, default=0, help='Plot results from validiation set (1) or test set (0, default).')
parser.add_argument('--overwrite', type=int, default=0, help='Will overwrite previous data if set to 1.')
arg = parser.parse_args()

# we care about starting performances.
# perf of annotators, starting classifiers
res_dir = arg.res_dir
model = arg.model

# dataset = f"{arg.dataset}_{exp_id}"
dataset = arg.dataset
last_seed = utils.get_last_seed(os.path.join(res_dir, dataset), f"{arg.dataset}_MC_")
seeds = list(range(last_seed+1))
print(f"Reading csvs from {res_dir}/{arg.dataset}/{arg.dataset}_EXPID_SEED/{model}_test_OVER/AGG.csv")
frames = []
aframes = []


if dataset == "ColoredMNIST":
    id_list = ['MC', 'ARGEN', 'AR', 'NOMC25', 'ARNOMC25']
elif dataset == "celeba":
    id_list = ['MC']
else:
    id_list = ['MC', 'ARGEN', 'ARCLA', 'NOMC25', 'ARNOMC25']

for exp_id in id_list:
    for seed in seeds:
        # print(seed)
        res_file = os.path.join(res_dir, dataset, f"{dataset}_{exp_id}_{seed}", f'{model}_test_overall.csv')
        # print(res_file)
        agg_file = os.path.join(res_dir, dataset, f"{dataset}_{exp_id}_{seed}", f'{model}_test_aggregated.csv')
        # print(agg_file)
        frame = pd.read_csv(res_file)
        aframe = pd.read_csv(agg_file)
        frame = remove_dup_runs(frame)
        aframe = remove_dup_runs(aframe)
        seed_col = [seed] * len(frame)
        frame['seed'] = seed_col
        aframe['seed'] = seed_col
        frames.append(frame)
        aframes.append(aframe)
data = pd.concat(frames, ignore_index = True)
agg_data = pd.concat(aframes, ignore_index=True)

over_mean = np.mean(data, axis=0)
over_stdev = np.std(data, axis=0)
agg_mean = np.mean(agg_data, axis=0)
agg_std = np.std(agg_data, axis=0)

print(f"Acc: {over_mean['Accuracy']} +- {over_stdev['Accuracy']}")
print(f"Acc diff: {agg_mean['difference_Accuracy']} +- {agg_std['difference_Accuracy']}")
print(f"DP diff: {agg_mean['difference_Selection rate']} +- {agg_std['difference_Selection rate']}")
# tprfpr_diffs = agg_mean[['difference_tpr', 'difference_fpr']]
# diffs = agg_std[['difference_tpr', 'difference_fpr']]
print(f"Eoods: {max(agg_mean['difference_tpr'], agg_mean['difference_fpr'])} +- {max(agg_std['difference_tpr'], agg_std['difference_fpr'])}")
total_count = over_mean['Count']
# print(agg_mean)
# print(over_mean)
# print(f"Count : {agg_mean['difference_Count'] / total_count}")