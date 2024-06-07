import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns
import utils
import pandas as pd
import argparse
import numpy as np

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'axes.titlepad': 5,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.linewidth': 1,
    'axes.labelpad': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.pad': 6,
    'ytick.major.pad': 6,
    'legend.title_fontsize': 16,
    'legend.fontsize': 14,
    # 'legend.frameon': False,
    'lines.markersize': 1.3,
    'legend.markerscale': 2,
    # 'legend.facecolor': 'white'
})

def remove_dup_runs(df):
    # repeated experiments are annoying. Use to keep only the most recent records in the file
    # get index for duplicate generations, remove these later occurances, keeping the first ones
    df = df.drop_duplicates(keep='last', subset=['generation'], ignore_index=True)
    return df


def plot_mus(muss, save_path):
    for i, mus in enumerate(muss):
        plt.figure()
        plt.hist(mus.flatten(), bins=150, label = f"mu{i}", alpha=0.5)
        plt.xlim(-3,3)
        plt.ylim(0, 100)
        plt.legend()
        plt.savefig(os.path.join(save_path, f"mus_{i}.pdf"), dpi=400)


def plot_smooth_mus(muss, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure() 
    for i, mus in enumerate(muss):
        if i%5 == 0:
            sns.kdeplot(mus.flatten(), bw_adjust=1, label=f"iteration {i}")
    plt.xlim(-3,3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"mus_iters.pdf"), dpi=400, bbox_inches='tight')

def init_save_dir(save_dir, overwrite):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:  # figures already exist, check before overwriting
        if overwrite == 0:
            print(f"Saving directory {save_dir} is not empty, stopping.")
            return False
    return True

def read_csvs(res_dir, dataset, exp_id, is_valid):
    if is_valid == 1:
        valid = "valid"
    else:
        valid = 'test'
    last_seed = utils.get_last_seed(res_dir, f"{dataset}_{exp_id}_")
    frames = []
    oframes = []
    if dataset == 'celeba':
        # seeds = [9, 1, 4, 5, 6, 7]
        # seeds = list(range(2,7))
        seeds = list(range(0, 5))
    else:
        seeds = list(range(last_seed + 1))
    for seed in seeds:
        res_file = os.path.join(res_dir, f"{dataset}_{exp_id}_{seed}", f'cla_{valid}_aggregated.csv')
        over_file = os.path.join(res_dir, f"{dataset}_{exp_id}_{seed}", f'cla_{valid}_overall.csv')
        try:
            frame = pd.read_csv(res_file)
            frame = remove_dup_runs(frame)
            oframe = pd.read_csv(over_file)
            oframe = remove_dup_runs(oframe)
            seed_col = [seed] * len(frame)
            frame['seed'] = seed_col
            oframe['seed'] = seed_col
            frames.append(frame)
            oframes.append(oframe)
        except:
            print(f'No {exp_id} file for seed {seed}')
    data = pd.concat(frames, ignore_index = True)
    overs = pd.concat(oframes, ignore_index=True)
    return data, overs

def read_csv(res_dir, dataset, exp_id, csv_name):
    last_seed = utils.get_last_seed(res_dir, f"{dataset}_{exp_id}_")
    frames = []
    # seeds = list(range(0, 5))
    for seed in seeds:
        res_file = os.path.join(res_dir, f"{dataset}_{exp_id}_{seed}", f'{csv_name}')
        try:
            frame = pd.read_csv(res_file)
            frame = remove_dup_runs(frame)
            seed_col = [seed] * len(frame)
            frame['seed'] = seed_col
            frames.append(frame)
        except:
            print(f"No csv: {res_file}")
    data = pd.concat(frames, ignore_index = True)
    return data


def generator_loss(data, overwrite, save_dir, xlim=40):
    print(data['generation'].value_counts())
    print(data['epoch'].value_counts())
    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    sns.lineplot(data, x='generation', y='gen_loss', ax=ax)
    ax.set_ylabel('$G_i$ loss wrt $G_{i-1}$')
    plt.xlim((0, xlim))
    plt.savefig(f'{save_dir}/gen_loss.pdf', dpi=400, bbox_inches='tight')
    plt.close()


def fair_over_generations(data, overs, save_dir, is_valid, overwrite, xlimit=40):
    if not init_save_dir(save_dir, overwrite):
        return
    
    if is_valid == 1:
        valid = "valid"
    else:
        valid = 'test'

    ### DP Diff ###
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='difference_Selection rate', ax=ax1, color='tab:red', legend=False)
    ax1.set_ylabel('Group selection rate difference', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()
    sns.lineplot(overs, x='generation', y='Selection rate', ax=ax2, color='tab:blue', legend=False)
    ax2.set_ylabel('Overall selection rate', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.title('Demographic Parity difference over generations')
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_dp_diff.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()

    ### DP Ratio ###
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='ratio_Selection rate', ax=ax1, color='tab:red', legend=False)
    ax1.set_ylabel('Group selection rate ratio', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # ax2 = ax1.twinx()
    # sns.lineplot(overs, x='generation', y='Selection rate', ax=ax2, color='tab:blue', legend=False)
    # ax2.set_ylabel('Overall selection rate', color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.title('Demographic Parity ratio over generations')
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_dp_ratio.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()

    ### MIN MAX SEL RATES ###
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='group_0_Selection rate', ax=ax1, color='tab:green', label=None)
    plt.axhline(y=.7, color='tab:green')  # TODO adjust for actual group selection rates
    ax1.set_ylabel('Group selection rates')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    # ax2 = ax1.twinx()
    sns.lineplot(data, x='generation', y='group_1_Selection rate', ax=ax1, color='tab:red')
    plt.axhline(y=.3, color='tab:red')
    # ax2.set_ylabel('Group min selection rate', color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.title('Group selection rate over generations')
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_grp_sel.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()

    # ### ACC ###
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='difference_Accuracy', ax=ax1, color='tab:red', legend=False)
    ax1.set_ylabel('Group accuracy difference', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax2 = ax1.twinx()
    sns.lineplot(overs, x='generation', y='Accuracy', ax=ax2, color='tab:blue', legend=False)
    ax2.set_ylabel('Overall accuracy', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.title('Group accuracy difference over generations')
    fig.tight_layout()
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_acc_diff.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    sns.lineplot(data, x='generation', y='group_0_Accuracy', ax=ax, color='tab:green', legend=False)
    sns.lineplot(data, x='generation', y='group_1_Accuracy', ax=ax, color='tab:red', legend=False)
    ax.set_ylabel('Accuracy')
    # plt.title('Group accuracies')
    plt.xlim((0, xlimit))
    plt.savefig(os.path.join(save_dir, f"{valid}_grp_acc.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()
    plt.close()

    ### FPR FNR TPR TNR ###
    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    sns.lineplot(data, x='generation', y='group_0_fpr', ax=ax, color='tab:green')
    sns.lineplot(data, x='generation', y='group_1_fpr', ax=ax, color='tab:red')
    ax.set_ylabel('Group FPRs')
    # plt.title('Group FPR over generations')
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_grp_fpr.pdf"), dpi=400, bbox_inches='tight')    
    plt.clf()
    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    sns.lineplot(data, x='generation', y='group_1_fnr', ax=ax, color='tab:red')
    sns.lineplot(data, x='generation', y='group_0_fnr', ax=ax, color='tab:green')
    ax.set_ylabel('Group FNRs')
    # plt.title('Group FNR over generations')
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_grp_fnr.pdf"), dpi=400, bbox_inches='tight')    
    plt.clf()
    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    sns.lineplot(data, x='generation', y='group_0_tpr', ax=ax, color='tab:green')
    sns.lineplot(data, x='generation', y='group_1_tpr', ax=ax, color='tab:red')
    ax.set_ylabel('Group TPRs')
    # plt.title('Group TPR over generations')
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_grp_tpr.pdf"), dpi=400, bbox_inches='tight')    
    plt.clf()
    plt.close()
    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    sns.lineplot(data, x='generation', y='group_1_tnr', ax=ax, color='tab:red')
    sns.lineplot(data, x='generation', y='group_0_tnr', ax=ax, color='tab:green')
    ax.set_ylabel('Group TNRs')
    # plt.title('Group TNR over generations')
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_grp_tnr.pdf"), dpi=400, bbox_inches='tight')    
    plt.clf()
    plt.close()

    ### EODDS DIFF ###
    tprfpr_diffs = data[['difference_tpr', 'difference_fpr']]
    eodds_diff = tprfpr_diffs.max(axis=1)
    data['eodds_diff'] = eodds_diff
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='eodds_diff', ax=ax1, color='tab:red', legend=False)
    ax1.set_ylabel('Equalized Odds Difference', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # ax2 = ax1.twinx()
    # sns.lineplot(overs, x='generation', y='Accuracy', ax=ax2, color='tab:blue', legend=False)
    # ax2.set_ylabel('Overall accuracy', color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.title('EOdds difference over generations')
    fig.tight_layout()
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_eodds_diff.pdf"), dpi=400, bbox_inches='tight')
    plt.close()

    ### EODDS RAT ###
    tprfpr_rats = data[['ratio_tpr', 'ratio_fpr']]
    data['eodds_ratio'] = tprfpr_rats.min(axis=1)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='eodds_ratio', ax=ax1, color='tab:red', legend=False)
    ax1.set_ylabel('Equalized Odds Ratio', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # plt.title('EOdds ratio over generations')
    fig.tight_layout()
    plt.xlim(left=0, right=xlimit)
    plt.savefig(os.path.join(save_dir, f"{valid}_eodds_ratio.pdf"), dpi=400, bbox_inches='tight')
    plt.close()


def population_stats(res_dir, dataset, exp_id, save_dir, overwrite, xlimit=40):
    if not init_save_dir(save_dir, overwrite):
        return
    data = read_csv(res_dir, dataset, exp_id, 'gen_pop_stats.csv')

    ### GROUP BALANCE ###
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='color_bal', ax=ax1, color='tab:red', legend=False)
    ax1.set_ylabel('Proportion of disadvantaged group', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # ax2 = ax1.twinx()
    # sns.lineplot(overs, x='generation', y='Selection rate', ax=ax2, color='tab:blue', legend=False)
    # ax2.set_ylabel('Overall selection rate', color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.title('Disadvantaged (red) group population in generated data')
    plt.xlim((0, xlimit))
    plt.savefig(os.path.join(save_dir, "gen_grp_bal.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()

    ### LABEL BAL ###
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='label_bal', ax=ax1, color='tab:red', legend=False)
    ax1.set_ylabel('Porportion of positive class', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # ax2 = ax1.twinx()
    # sns.lineplot(overs, x='generation', y='Selection rate', ax=ax2, color='tab:blue', legend=False)
    # ax2.set_ylabel('Overall selection rate', color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.title('Proportion of positive (beneficial) label in generated data')
    plt.xlim((0, xlimit))
    plt.savefig(os.path.join(save_dir, "gen_lab_bal.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()


def sensitive_confidence(res_dir, dataset, exp_id, save_dir, overwrite, xlimit=40):
    if not init_save_dir(save_dir, overwrite):
        return
    
    data = read_csv(res_dir, dataset, exp_id, 'confidence_sens.csv')

    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='proba_0', ax=ax, color='tab:green', legend=False)
    sns.lineplot(data, x='generation', y='proba_1', ax=ax, color='tab:red', legend=False)
    ax.set_ylabel('Proba confidence margin')
    
    plt.xlim(right=xlimit)
    plt.savefig(os.path.join(save_dir, "sens_conf.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()


def ar_to_resample(res_dir, dataset, exp_id, save_dir, use_reparation, batch_size, rep_budget, overwrite, xlimit=40):
    budget = rep_budget
    
    if not init_save_dir(save_dir, overwrite):
        return
    
    dataset = f"{dataset}_{exp_id}"
    last_seed = utils.get_last_seed(res_dir, f"{dataset}_")
    seeds = list(range(last_seed+1))
    seeds = list(range(0, 5))

    print(f"Reading csvs from {res_dir}/{dataset}_SEED/ar_{use_reparation}_batches.csv")
    print(f"Saving figures to {save_dir}")

    frames = []
    for seed in seeds:
        try:
            res_file = os.path.join(res_dir, f"{dataset}_{seed}", f'ar_{use_reparation}_batches.csv')
            frame = pd.read_csv(res_file)
        except:
            res_file = os.path.join(res_dir, f"{dataset}_{seed}", f'ar_batches.csv')
            frame = pd.read_csv(res_file)
        frame = remove_dup_runs(frame)
        seed_col = [seed] * len(frame)
        frame['seed'] = seed_col
        frames.append(frame)

    data = pd.concat(frames, ignore_index = True)
    data['prop_resampled'] = data['resampled'] / batch_size
    num_generations = data['generation'].max() + 1

    # form dataframe
    num_generations = 8
    gens = list(range(num_generations)) * len(seeds) 
    seed_list = []
    prop_resampled = []
    for seed in seeds:
        seed_list += [seed] * num_generations
        for gen in range(num_generations):
            to_avg = data.loc[ (data['seed']==seed) & (data['generation']==gen) ]
            avged = np.mean(to_avg['prop_resampled'])
            prop_resampled.append(avged)
    dct = {'generation': gens, 'seed': seed_list, 'prop_resampled': prop_resampled}
    avgd_df = pd.DataFrame(dct)

    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    ax.set_ylabel('Resampled proprtion')
    sns.lineplot(avgd_df, x='generation', y='prop_resampled')
    # plt.title('Resampled proportion of batches per generation')
    plt.savefig(os.path.join(save_dir, "to_resample.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()


def ar_categories(res_dir, dataset, exp_id, save_dir, use_reparation, batch_size, rep_budget, overwrite):
    batch_size = batch_size + rep_budget
    print(batch_size)
    if not init_save_dir(save_dir, overwrite):
        return
    
    dataset = f"{dataset}_{exp_id}"
    last_seed = utils.get_last_seed(res_dir, f"{dataset}_")
    seeds = list(range(last_seed+1))
    seeds = list(range(0, 5))
    print(f"Reading csvs from {res_dir}/{dataset}_SEED/ar_{use_reparation}_batches.csv")
    print(f"Saving figures to {save_dir}")
    frames = []
    for seed in seeds:
        try:
            res_file = os.path.join(res_dir, f"{dataset}_{seed}", f'ar_{use_reparation}_batches.csv')
            frame = pd.read_csv(res_file)
        except:
            res_file = os.path.join(res_dir, f"{dataset}_{seed}", f'ar_batches.csv')
            frame = pd.read_csv(res_file)
        seed_col = [seed] * len(frame)
        frame['seed'] = seed_col
        frames.append(frame)
    data = pd.concat(frames, ignore_index = True)
    num_generations = data['generation'].max() + 1

    # form dataframe
    gens = list(range(num_generations)) * len(seeds)
    seed_list = []
    categories = []
    for seed in seeds:
        seed_list += [seed] * num_generations
        for gen in range(num_generations):
            to_avg = data.loc[ (data['seed']==seed) & (data['generation']==gen) ]
            avged = np.mean(to_avg[['c0g0','c0g1','c1g0','c1g1']], axis=0) #/ batch_size * 100).round(2)
            row = (gen, seed, avged[0], avged[1], avged[2], avged[3])
            categories.append(row)
    avgd_df = pd.DataFrame(categories, columns=['generation','seed','c0g0','c0g1','c1g0','c1g1'])
    # get averages and stdevs over the seeds
    avges = []
    stdevs = []
    for gen in range(num_generations):
        data = avgd_df.loc[ avgd_df['generation']==gen ]
        avged = (np.mean(data[['c0g0','c0g1','c1g0','c1g1']], axis=0) / batch_size * 100).round(2)
        stdeved = (np.std(data[['c0g0','c0g1','c1g0','c1g1']], axis=0) / batch_size * 100).round(2)
        row = (avged[0], avged[1], avged[2], avged[3])
        avges.append(row)
    plottable = pd.DataFrame(avges, columns=['c0g0','c0g1','c1g0','c1g1'])

    fig, ax = plt.subplots()
    plottable.plot.bar(stacked=True, ax=ax, color=['lightgreen', 'lightcoral', 'green', 'red'])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.xlabel('Generations')
    plt.xticks(np.arange(0, num_generations, step=5))
    plt.yticks([0, 25, 50, 75, 100])
    plt.ylim((0, 100))
    plt.ylabel('AR sampling categories %')
    legend = ax.legend(labels=['Majoritized, Class 0', 'Minoritized, Class 0', 'Majoritized, Class 1', 'Minoritized, Class 1'], \
                       title='Demographic Category')
    legend.get_frame().set_facecolor('white')
    plt.savefig(os.path.join(save_dir, "resample_cats.pdf"), dpi=400, bbox_inches='tight')
    plt.clf()
    

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ColoredMNIST', help='Dataset')
    parser.add_argument('--exp-id-noseed', type=str, default='debugging', help='Experiment name, do not include seed.')
    parser.add_argument('--use-reparation', type=str, default=None, help='cal, gen, both, or None (default) for which model has ar')
    parser.add_argument('--save-dir', type=str, default='./figs', help='Overall directory to save figures into.')
    parser.add_argument('--res-dir', type=str, default='./results', help='Overall directory where result csvs are stored.')
    parser.add_argument('--is-valid', type=int, default=0, help='Plot results from validiation set (1) or test set (0, default).')
    parser.add_argument('--overwrite', type=int, default=0, help='Will overwrite previous data if set to 1.')
    parser.add_argument('--rep-budget', type=int, default=0, help='size of rep budget')
    parser.add_argument('--batch-size', type=int, default=0, help='batch size of repaired model')

    arg = parser.parse_args()

    save_path = os.path.join(arg.save_dir, f"{arg.dataset}/{arg.dataset}_{arg.exp_id_noseed}")
    arg.res_dir = os.path.join(arg.res_dir, f"{arg.dataset}")
    
    agg, over = read_csvs(arg.res_dir, arg.dataset, arg.exp_id_noseed, arg.is_valid)
    fair_over_generations(agg, over, save_path, arg.is_valid, arg.overwrite, xlimit=15)
    population_stats(arg.res_dir, arg.dataset, arg.exp_id_noseed, save_path, arg.overwrite, xlimit=15)
    sensitive_confidence(arg.res_dir, arg.dataset, arg.exp_id_noseed, save_path, arg.overwrite, xlimit=15)
    if arg.use_reparation not in [None, 'None']:
        ar_to_resample(arg.res_dir, arg.dataset, arg.exp_id_noseed, save_path, arg.use_reparation, arg.batch_size, arg.rep_budget, arg.overwrite)
        ar_categories(arg.res_dir, arg.dataset, arg.exp_id_noseed, save_path, arg.use_reparation, arg.batch_size, arg.rep_budget, arg.overwrite)

    data = read_csv(arg.res_dir, arg.dataset, arg.exp_id_noseed, 'gen_loss.csv')
    generator_loss(data, 1, save_path, xlim=40)

"""
How to call for our SeqCla with AR results:

python plotting.py --dataset ColoredMNIST --exp-id-noseed ARNOMC25 --save-dir ./replotted/comparisons --res-dir ./results --is-valid 1 --overwrite 1 \
                    --use-reparation cla --rep-budget 64 --batch-size 256


python plotting.py --dataset SVHN --exp-id-noseed ARNOMC25 --save-dir ./replotted/comparisons --res-dir ./results --is-valid 1 --overwrite 1 \
                    --use-reparation cla --rep-budget 8 --batch-size 32

python plotting.py --dataset celeba --exp-id-noseed ARNOMC25 --save-dir ./replotted/comparisons --res-dir ./results --is-valid 1 --overwrite 1 \
                    --use-reparation cla --rep-budget 32 --batch-size 128
"""
