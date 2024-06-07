import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns
import utils
import pandas as pd
import argparse

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.titlepad': 5,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1,
    'axes.labelpad': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.pad': 4,
    'ytick.major.pad': 4,
    'legend.title_fontsize': 14,
    'legend.frameon': False,
    'lines.markersize': 1.3,
    'legend.markerscale': 2
})

def remove_dup_runs(df):
    # repeated experiments are annoying. Use to keep only the most recent records in the file
    # get index for duplicate generations, remove these later occurances, keeping the first ones
    df = df.drop_duplicates(keep='last', subset=['generation'], ignore_index=True)
    return df


def syn_ablation(res_dir, abl_name, is_valid=False):
    # get dataframe with column for perc, column for seed  
    max_seed = utils.get_last_seed(res_dir, f"{dataset}_{abl_name}0_")
    if max_seed == -1:
        max_seed = utils.get_last_seed(res_dir, f"{dataset}_{abl_name}50_")
    if is_valid == 1:
        valid = "valid"
    else:
        valid = 'test'

    frames = []
    oframes = []
    gframes = []
    for i in [0, 20, 40, 60, 80, 90]:
    # for i in [0, 2, 4, 6, 8, 9]:
    # for i in range(50, 100, 10):
        for seed in range(max_seed+1):
            # each file corresponds to one perc and one seed
            res_file = os.path.join(res_dir, f"{dataset}_{abl_name}{i}_{seed}", f'cla_{valid}_aggregated.csv')
            over_file = os.path.join(res_dir, f"{dataset}_{abl_name}{i}_{seed}", f'cla_{valid}_overall.csv')
            gpop = os.path.join(res_dir, f"{dataset}_{abl_name}{i}_{seed}", f'gen_pop_stats.csv')
            frame = pd.read_csv(res_file)
            frame = remove_dup_runs(frame)
            oframe = pd.read_csv(over_file)
            oframe = remove_dup_runs(oframe)
            gframe = pd.read_csv(gpop)
            gframe = remove_dup_runs(gframe)
            # add seed and perc collumns
            seed_col = [seed] * len(frame)
            frame['seed'] = seed_col
            frame['abl'] = [i] * len(frame)
            oframe['seed'] = seed_col
            oframe['abl'] = [i] * len(oframe)
            seed_col = [seed] * len(gframe)
            gframe['seed'] = seed_col
            gframe['abl'] = [i] * len(gframe)
            frames.append(frame)
            oframes.append(oframe)
            gframes.append(gframe)

    data = pd.concat(frames, ignore_index = True)
    overs = pd.concat(oframes, ignore_index=True)
    gs = pd.concat(gframes, ignore_index=True)
    return data, overs, gs


def acc(aggs, overs, save_dir, is_valid=False, overwrite=0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:  # figures already exist, check before overwriting
        if overwrite == 0:
            print(f"Saving directory {save_dir} is not empty, stopping.")
            return
    if is_valid == 1:
        valid = "valid"
    else:
        valid = 'test'
        
    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(overs, x='generation', y='Accuracy', hue='abl', ax=ax)
    ax.set_ylabel('Accuracy')
    plt.title('Accuracy over generations synthetic ablation')
    plt.savefig(os.path.join(save_dir, f"{valid}_accuracy.pdf"), dpi=400)
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(aggs, x='generation', y='difference_Accuracy', hue='abl', ax=ax)
    ax.set_ylabel('Accuracy difference')
    plt.title('Accuracy diff over generations synthetic ablation')
    plt.savefig(os.path.join(save_dir, f"{valid}_acc_diff.pdf"), dpi=400)
    plt.clf()
    plt.close()
    return

def alls(data, overs, save_dir, legend_title, is_valid=False, overwrite=0):
    num_abls = len(data['abl'].value_counts())
    reds = sns.color_palette("rocket_r", num_abls)
    blues = sns.color_palette("mako_r", num_abls)
    greens = sns.color_palette("BuGn", num_abls)

    # data['abl'] = 10 * data['abl']
    # overs['abl'] = 10 * overs['abl']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:  # figures already exist, check before overwriting
        if overwrite == 0:
            print(f"Saving directory {save_dir} is not empty, stopping.")
            return
    if is_valid == 1:
        valid = "valid"
    else:
        valid = 'test'
    ### DP Diff ###
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='difference_Selection rate', ax=ax1, palette=reds, hue='abl')
    ax1.set_ylabel('Demographic Parity Difference', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # ax2 = ax1.twinx()
    # sns.lineplot(overs, x='generation', y='Selection rate', ax=ax2, palette=blues, hue='abl')
    # ax2.set_ylabel('Overall selection rate', color='tab:blue')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.title('Demographic Parity difference over generations')
    plt.legend(title=legend_title)
    plt.savefig(os.path.join(save_dir, f"{valid}_dp_diff.pdf"), dpi=400)
    plt.clf()
    plt.close()

    '''
    ### DP Ratio ###
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('Generations')
    # # ax1.set_xticks(list(range(50)))
    # sns.lineplot(data, x='generation', y='ratio_Selection rate', ax=ax1, palette=reds, hue='abl')
    # ax1.set_ylabel('Group selection rate ratio', color='tab:red')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    # plt.title('Demographic Parity ratio over generations')
    # plt.savefig(os.path.join(save_dir, f"{valid}_dp_ratio.pdf"), dpi=400)
    # plt.clf()
    # plt.close()

    ### MIN MAX SEL RATES ###
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('Generations')
    # # ax1.set_xticks(list(range(50)))
    # sns.lineplot(data, x='generation', y='group_max_Selection rate', ax=ax1, palette=greens, label=None, hue='abl')
    # ax1.set_ylabel('Group selection rates')
    # # ax1.tick_params(axis='y', labelcolor='tab:red')
    # # ax2 = ax1.twinx()
    # sns.lineplot(data, x='generation', y='group_min_Selection rate', ax=ax1, palette=reds, linestyle='--', hue='abl')
    # plt.axhline(y=.3, color='tab:red')
    # # ax2.set_ylabel('Group min selection rate', color='tab:blue')
    # # ax2.tick_params(axis='y', labelcolor='tab:blue')
    # plt.title('Group selection rate over generations')
    # plt.savefig(os.path.join(save_dir, f"{valid}_grp_sel.pdf"), dpi=400)
    # plt.clf()
    # plt.close()
    '''
    # ### ACC ###
    fig, ax1 = plt.subplots()
    sns.lineplot(data, x='generation', y='difference_Accuracy', ax=ax1, palette=reds, hue='abl')
    ax1.set_ylabel('Accuracy Difference', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xlabel('Generations')
    plt.legend(title=legend_title)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{valid}_acc_diff.pdf"), dpi=400)
    plt.clf()
    plt.close()

    fig, ax1 = plt.subplots()
    sns.lineplot(overs, x='generation', y='Accuracy', ax=ax1, palette=reds, hue='abl')
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Accuracy', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    plt.legend(title=legend_title)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{valid}_acc.pdf"), dpi=400)
    plt.clf()
    plt.close()
    '''
    ### FPR FNR TPR TNR ###
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Generations')
    # sns.lineplot(data, x='generation', y='group_max_fpr', ax=ax, linestyle='--', palette=greens, hue='abl')
    # sns.lineplot(data, x='generation', y='group_min_fpr', ax=ax, palette=reds, hue='abl')
    # ax.set_ylabel('Group FPRs')
    # plt.title('Group FPR over generations')
    # plt.savefig(os.path.join(save_dir, f"{valid}_grp_fpr.pdf"), dpi=400)    
    # plt.clf()
    # plt.close()
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Generations')
    # sns.lineplot(data, x='generation', y='group_max_fnr', ax=ax, palette=reds, hue='abl')
    # sns.lineplot(data, x='generation', y='group_min_fnr', ax=ax, linestyle='--', palette=greens, hue='abl')
    # ax.set_ylabel('Group FNRs')
    # plt.title('Group FNR over generations')
    # plt.savefig(os.path.join(save_dir, f"{valid}_grp_fnr.pdf"), dpi=400)    
    # plt.clf()
    # plt.close()
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Generations')
    # sns.lineplot(data, x='generation', y='group_max_tpr', ax=ax, linestyle='--', palette=greens, hue='abl')
    # sns.lineplot(data, x='generation', y='group_min_tpr', ax=ax, palette=reds, hue='abl')
    # ax.set_ylabel('Group TPRs')
    # plt.title('Group TPR over generations')
    # plt.savefig(os.path.join(save_dir, f"{valid}_grp_tpr.pdf"), dpi=400)    
    # plt.clf()
    # plt.close()
    # fig, ax = plt.subplots()
    # ax.set_xlabel('Generations')
    # sns.lineplot(data, x='generation', y='group_max_tnr', ax=ax, palette=reds, hue='abl')
    # sns.lineplot(data, x='generation', y='group_min_tnr', ax=ax, linestyle='--', palette=greens, hue='abl')
    # ax.set_ylabel('Group TNRs')
    # plt.title('Group TNR over generations')
    # plt.savefig(os.path.join(save_dir, f"{valid}_grp_tnr.pdf"), dpi=400)    
    # plt.clf()
    # plt.close()
    '''
    ### EODDS DIFF ###
    tprfpr_diffs = data[['difference_tpr', 'difference_fpr']]
    eodds_diff = tprfpr_diffs.max(axis=1)
    data['eodds_diff'] = eodds_diff
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='eodds_diff', ax=ax1, palette=reds, hue='abl')
    ax1.set_ylabel('Equalized Odds Difference', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    # plt.title('EOdds difference over generations')
    plt.legend(title=legend_title)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{valid}_eodds_diff.pdf"), dpi=400)
    plt.close()

    ### EODDS RAT ###
    # tprfpr_rats = data[['ratio_tpr', 'ratio_fpr']]
    # data['eodds_ratio'] = tprfpr_rats.min(axis=1)
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('Generations')
    # # ax1.set_xticks(list(range(50)))
    # sns.lineplot(data, x='generation', y='eodds_ratio', ax=ax1, palette=reds, hue='abl')
    # ax1.set_ylabel('Equalized Odds Ratio', color='tab:red')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    # plt.title('EOdds ratio over generations')
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_dir, f"{valid}_eodds_ratio.pdf"), dpi=400)
    # plt.close()

def pop_abl(data, save_dir, legend_title, overwrite=0):
    num_abls = len(data['abl'].value_counts())
    reds = sns.color_palette("rocket_r", num_abls)

    # data['abl'] = 10 * data['abl']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:  # figures already exist, check before overwriting
        if overwrite == 0:
            print(f"Saving directory {save_dir} is not empty, stopping.")
            return
    
    fig, ax1 = plt.subplots()
    sns.lineplot(data, x='generation', y='label_bal', ax=ax1, palette=reds, hue='abl')
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Beneficial Class', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    plt.legend(title=legend_title)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"label_bal.pdf"), dpi=400)
    plt.clf()
    plt.close()

    fig, ax1 = plt.subplots()
    sns.lineplot(data, x='generation', y='color_bal', ax=ax1, palette=reds, hue='abl')
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Minoritized Group', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    plt.legend(title=legend_title)
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, f"group_bal.pdf"), dpi=400)
    plt.clf()
    plt.close()
    


def population_stats(res_dir, exp_id, save_dir, overwrite):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:  # figures already exist, check before overwriting
        if overwrite == 0:
            print(f"Saving directory {save_dir} is not empty, stopping.")
            return

    dataset = f"ColoredMNIST_{exp_id}"
    last_seed = utils.get_last_seed(res_dir, f"{dataset}_")
    seeds = list(range(last_seed+1))

    print(f"Reading csvs from {res_dir}{dataset}_SEED/gen_pop_stats.csv")
    print(f"Saving figures to {save_dir}")

    frames = []
    for seed in seeds:
        res_file = os.path.join(res_dir, f"{dataset}_{seed}", f'gen_pop_stats.csv')
        frame = pd.read_csv(res_file)
        frame = remove_dup_runs(frame)
        seed_col = [seed] * len(frame)
        frame['seed'] = seed_col
        frames.append(frame)

    data = pd.concat(frames, ignore_index = True)

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
    plt.title('Disadvantaged (red) group population in generated data')
    plt.savefig(os.path.join(save_dir, "gen_grp_bal.pdf"), dpi=400)
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
    plt.title('Proportion of positive (beneficial) label in generated data')
    plt.savefig(os.path.join(save_dir, "gen_lab_bal.pdf"), dpi=400)
    plt.clf()

def sensitive_confidence(data, overs, save_dir, overwrite):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:  # figures already exist, check before overwriting
        if overwrite == 0:
            print(f"Saving directory {save_dir} is not empty, stopping.")
            return
    
    dataset = f"ColoredMNIST_{exp_id}"
    last_seed = utils.get_last_seed(res_dir, f"{dataset}_")
    seeds = list(range(last_seed+1))

    print(f"Reading csvs from {res_dir}{dataset}_SEED/confidence_sens.csv")
    print(f"Saving figures to {save_dir}")

    frames = []
    for seed in seeds:
        res_file = os.path.join(res_dir, f"{dataset}_{seed}", f'confidence_sens.csv')
        frame = pd.read_csv(res_file)
        frame = remove_dup_runs(frame)
        seed_col = [seed] * len(frame)
        frame['seed'] = seed_col
        frames.append(frame)

    data = pd.concat(frames, ignore_index = True)

    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(data, x='generation', y='proba_0', ax=ax, color='tab:green', legend=False)
    sns.lineplot(data, x='generation', y='proba_1', ax=ax, color='tab:red', legend=False)
    ax.set_ylabel('Proba confidence margin')
    
    plt.title('Confidence [0, .5] for predicintg sentive attr of generated data')
    plt.savefig(os.path.join(save_dir, "sens_conf.pdf"), dpi=400)
    plt.clf()

def dp_diff(aggs, overs, save_dir, is_valid=False, overwrite=0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:  # figures already exist, check before overwriting
        if overwrite == 0:
            print(f"Saving directory {save_dir} is not empty, stopping.")
            return
    if is_valid == 1:
        valid = "valid"
    else:
        valid = 'test'
        
    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    # ax1.set_xticks(list(range(50)))
    sns.lineplot(aggs, x='generation', y='difference_Selection rate', hue='abl', ax=ax)
    ax.set_ylabel('DP difference')
    plt.title('DP diff over generations synthetic ablation')
    plt.savefig(os.path.join(save_dir, f"{valid}_dp_diff.pdf"), dpi=400)
    plt.clf()
    plt.close()
    return


if __name__ == '__main__':
    dataset = 'ColoredMNIST'
    # perc = 70
    # seed = 0
    valid = True
    abl_name = 'syn'
    results_dir = f'./results/syn_ColoredMNIST'
    agg, overall, gs = syn_ablation(res_dir=results_dir, abl_name=abl_name, is_valid=valid)
    acc(agg, overall, save_dir=f'./figs/{dataset}_{abl_name}', overwrite=1)
    dp_diff(agg, overall, save_dir=f'./figs/{dataset}_{abl_name}', overwrite=1)
    alls(agg, overall, save_dir=f'./figs/{dataset}/{dataset}_{abl_name}', legend_title='% Synthetic Data', overwrite=1, is_valid=valid)
    pop_abl(gs, save_dir=f'./figs/{dataset}/{dataset}_{abl_name}', legend_title='% Synthetic Data', overwrite=1)
    population_stats(results_dir, abl_name, save_dir=f'./figs/{dataset}_{abl_name}', overwrite=1)
