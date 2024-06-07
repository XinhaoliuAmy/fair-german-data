import utils
import argparse
import os
import sys
import utils
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


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
    'legend.frameon': False,
    'lines.markersize': 1.3,
    'legend.markerscale': 2,
    'legend.facecolor': 'white'
})

green = 'tab:green'
red = 'tab:red'


def remove_dup_runs(df):
    # repeated experiments are annoying. Use to keep only the most recent records in the file
    # get index for duplicate generations, remove these later occurances, keeping the first ones
    df = df.drop_duplicates(keep='last', subset=[
                            'generation'], ignore_index=True)
    return df


def read_csvs(res_dir, dataset, exp_id, valid, read_pop_stats):
    last_seed = utils.get_last_seed(res_dir, f"{dataset}_{exp_id}_")
    frames = []
    oframes = []
    pop_frames = []
    for seed in range(last_seed + 1):
        res_file = os.path.join(
            res_dir, f"{dataset}_{exp_id}_{seed}", f'cla_{valid}_aggregated.csv')
        over_file = os.path.join(
            res_dir, f"{dataset}_{exp_id}_{seed}", f'cla_{valid}_overall.csv')
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
            if read_pop_stats:
                pop_file = os.path.join(
                    res_dir, f"{dataset}_{exp_id}_{seed}", f'gen_pop_stats.csv')
                pframe = pd.read_csv(pop_file)
                pframe = remove_dup_runs(pframe)
                seed_col = [seed] * len(pframe)
                pframe['seed'] = seed_col
                pop_frames.append(pframe)
        except:
            print(f'No {exp_id} file for seed {seed}')
    data = pd.concat(frames, ignore_index=True)
    overs = pd.concat(oframes, ignore_index=True)
    if read_pop_stats:
        pop_frames = pd.concat(pop_frames, ignore_index=True)
    return data, overs, pop_frames


def read_batches(res_dir, dataset, exp_id, use_reparation):
    last_seed = utils.get_last_seed(res_dir, f"{dataset}_{exp_id}_")
    frames = []
    for seed in range(last_seed + 1):
        res_file = os.path.join(
            res_dir, f"{dataset}_{exp_id}_{seed}", f'ar_{use_reparation}_batches.csv')
        try:
            frame = pd.read_csv(res_file)
            frame = remove_dup_runs(frame)
            seed_col = [seed] * len(frame)
            frame['seed'] = seed_col
            frames.append(frame)
        except:
            print(f'No {exp_id} file for seed {seed}')
    data = pd.concat(frames, ignore_index=True)
    return data


def arnomc(agg_ar, agg_no, save_dir, compare_name, xlim=40):
    sns.lineplot(agg_ar, x='generation', y='difference_Accuracy',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(agg_no, x='generation', y='difference_Accuracy',
                 color=red, legend=False)
    plt.ylabel('Accuracy difference')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4)]
    plt.legend(legend_elems, ['Reparation', 'No reparation'])
    plt.savefig(f'{save_dir}/{compare_name}_acc_diff.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()

    sns.lineplot(agg_ar, x='generation', y='difference_Selection rate',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(agg_no, x='generation',
                 y='difference_Selection rate', color=red, legend=False)
    plt.ylabel('Demographic Parity Diff')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4)]
    plt.legend(legend_elems, ['Reparation', 'No reparation'])
    # plt.legend(legend_elems, ['Sequential', 'Non-sequential'])
    plt.savefig(f'{save_dir}/{compare_name}_dp.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()

    tprfpr_diffs = agg_ar[['difference_tpr', 'difference_fpr']]
    eodds_diff = tprfpr_diffs.max(axis=1)
    agg_ar['eodds_diff'] = eodds_diff
    tprfpr_diffs = agg_no[['difference_tpr', 'difference_fpr']]
    eodds_diff = tprfpr_diffs.max(axis=1)
    agg_no['eodds_diff'] = eodds_diff
    sns.lineplot(agg_ar, x='generation', y='eodds_diff',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(agg_no, x='generation', y='eodds_diff',
                 color=red, legend=False)
    plt.ylabel('Equalized Odds Diff')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4)]
    plt.legend(legend_elems, ['Reparation', 'No reparation'])
    plt.savefig(f'{save_dir}/{compare_name}_eodds.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()


def arnomcoveralls(over_ar, over_sq, save_dir, compare_name, xlim=40):
    sns.lineplot(over_ar, x='generation', y='Accuracy',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(over_sq, x='generation',
                 y='Accuracy', color=red, legend=False)
    plt.ylabel('Accuracy')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4)]
    # plt.legend(legend_elems, ['Sequential', 'Non-sequential'])
    plt.legend(legend_elems, ['Reparation', 'No reparation'])
    plt.savefig(f'{save_dir}/{compare_name}_acc.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()

    sns.lineplot(over_ar, x='generation', y='Selection rate',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(over_sq, x='generation',
                 y='Selection rate', color=red, legend=False)
    plt.ylabel('Selection Rate')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4)]
    # plt.legend(legend_elems, ['Sequential', 'Non-sequential'])
    plt.legend(legend_elems, ['Reparation', 'No reparation'])
    plt.savefig(f'{save_dir}/{compare_name}_sel_rate.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()


def armc(agg_ar, agg_sq, agg_gar, save_dir, compare_name, xlim=40, shade=15):
    sns.lineplot(agg_ar, x='generation', y='difference_Selection rate',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(agg_sq, x='generation',
                 y='difference_Selection rate', color=red, legend=False)
    sns.lineplot(agg_gar, x='generation', y='difference_Selection rate',
                 color='tab:purple', legend=False, linestyle='dotted')
    plt.ylabel('Demographic Parity Diff')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color='tab:purple', lw=4, linestyle='dotted')]
    plt.legend(legend_elems, ['Classifier reparation',
               'No reparation', 'Generator reparation'])
    plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=.3)
    plt.savefig(f'{save_dir}/{compare_name}_dp.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()

    sns.lineplot(agg_ar, x='generation', y='difference_Accuracy',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(agg_sq, x='generation', y='difference_Accuracy',
                 color=red, legend=False)
    sns.lineplot(agg_gar, x='generation', y='difference_Accuracy',
                 color='tab:purple', legend=False, linestyle='dotted')
    plt.ylabel('Accuracy Diff')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color='tab:purple', lw=4, linestyle='dotted')]
    plt.legend(legend_elems, ['Classifier reparation',
               'No reparation', 'Generator reparation'])
    plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=.3)
    plt.savefig(f'{save_dir}/{compare_name}_acc_diff.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()

    tprfpr_diffs = agg_ar[['difference_tpr', 'difference_fpr']]
    eodds_diff = tprfpr_diffs.max(axis=1)
    agg_ar['eodds_diff'] = eodds_diff
    tprfpr_diffs = agg_gar[['difference_tpr', 'difference_fpr']]
    eodds_diff = tprfpr_diffs.max(axis=1)
    agg_gar['eodds_diff'] = eodds_diff
    tprfpr_diffs = agg_sq[['difference_tpr', 'difference_fpr']]
    eodds_diff = tprfpr_diffs.max(axis=1)
    agg_sq['eodds_diff'] = eodds_diff
    sns.lineplot(agg_ar, x='generation', y='eodds_diff',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(agg_sq, x='generation', y='eodds_diff',
                 color=red, legend=False)
    sns.lineplot(agg_gar, x='generation', y='eodds_diff',
                 color='tab:purple', legend=False, linestyle='dotted')
    plt.ylabel('Equalized Odds Diff')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color='tab:purple', lw=4, linestyle='dotted')]
    plt.legend(legend_elems, ['Classifier reparation',
               'No reparation', 'Generator reparation'])
    plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=.3)
    plt.savefig(f'{save_dir}/{compare_name}_eodds.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()


def aroveralls(over_ar, over_sq, over_gar, save_dir, compare_name, xlim=40, shade=15):
    sns.lineplot(over_ar, x='generation', y='Accuracy',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(over_sq, x='generation',
                 y='Accuracy', color=red, legend=False)
    sns.lineplot(over_gar, x='generation', y='Accuracy',
                 color='tab:purple', legend=False, linestyle='dotted')
    plt.ylabel('Accuracy')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color='tab:purple', lw=4, linestyle='dotted')]
    plt.legend(legend_elems, ['Classifier reparation',
               'No reparation', 'Generator reparation'])
    plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=.3)
    plt.savefig(f'{save_dir}/{compare_name}_acc.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()

    sns.lineplot(over_ar, x='generation', y='Selection rate',
                 color=green, legend=False, linestyle='dashed')
    sns.lineplot(over_sq, x='generation',
                 y='Selection rate', color=red, legend=False)
    sns.lineplot(over_gar, x='generation', y='Selection rate',
                 color='tab:purple', legend=False, linestyle='dotted')
    plt.ylabel('Selection Rate')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color='tab:purple', lw=4, linestyle='dotted')]
    plt.legend(legend_elems, ['Classifier reparation',
               'No reparation', 'Generator reparation'])
    plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=.3)
    plt.savefig(f'{save_dir}/{compare_name}_sel_rate.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()


def arpop(pop_ar, pop_sq, pop_gar, save_dir, compare_name, xlim=40, shade=15):
    sns.lineplot(pop_ar, x='generation', y='label_bal',
                 color='tab:green', legend=False, linestyle='dashed')
    sns.lineplot(pop_sq, x='generation', y='label_bal',
                 color='tab:red', legend=False)
    sns.lineplot(pop_gar, x='generation', y='label_bal',
                 color='tab:purple', legend=False, linestyle='dotted')
    plt.ylabel('Beneficial Class')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color='tab:purple', lw=4, linestyle='dotted')]
    plt.legend(legend_elems, ['Classifier reparation',
               'No reparation', 'Generator reparation'])
    plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=.3)
    plt.savefig(f'{save_dir}/{compare_name}_label_bal.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()

    sns.lineplot(pop_ar, x='generation', y='color_bal',
                 color='tab:green', legend=False, linestyle='dashed')
    sns.lineplot(pop_sq, x='generation', y='color_bal',
                 color='tab:red', legend=False)
    sns.lineplot(pop_gar, x='generation', y='color_bal',
                 color='tab:purple', legend=False, linestyle='dotted')
    plt.ylabel('Minoritized Group')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color='tab:purple', lw=4, linestyle='dotted')]
    plt.legend(legend_elems, ['Classifier reparation',
               'No reparation', 'Generator reparation'])
    plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=.3)
    plt.savefig(f'{save_dir}/{compare_name}_color_bal.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()


def rates(agg_ar, agg_sq, save_dir, compare_name, xlim=40):
    sns.lineplot(agg_ar, x='generation', y='difference_fpr',
                 color=green, legend=False)
    sns.lineplot(agg_sq, x='generation', y='difference_fpr',
                 color=red, legend=False)
    plt.ylabel('FPR Group Diff')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    plt.savefig(f'{save_dir}/{compare_name}_fpr_diff.pdf',
                dpi=400, bbox_inches='tight')
    plt.close()
    pass


def resampled_prop(res_dir, dataset, exp_id, save_dir, xlim=40):
    if dataset == "SVHN":
        bs = 128
        bscla = 32
    elif dataset == "ColoredMNIST":
        bs = 256
        bscla = 256
    else:
        bs = 64
        bscla = 128
    data = read_batches(res_dir, dataset, exp_id, 'gen')
    datacla = read_batches(res_dir, dataset, exp_id, 'cla')

    data['prop_resampled'] = data['resampled'] / bs
    datacla['prop_resampled'] = datacla['resampled'] / bscla
    num_generations = data['generation'].max() + 1

    # form dataframe
    seeds = 10
    gens = list(range(num_generations)) * len(seeds)
    seed_list = []
    prop_resampled = []
    sl = []
    pr = []
    for seed in seeds:
        seed_list += [seed] * num_generations
        for gen in range(num_generations):
            to_avg = data.loc[(data['seed'] == seed) &
                              (data['generation'] == gen)]
            to_avgcla = datacla.loc[(datacla['seed'] == seed) & (
                datacla['generation'] == gen)]

            avged = np.mean(to_avg['prop_resampled'])
            avgedcla = np.mean(to_avgcla['prop_resampled'])

            prop_resampled.append(avged)
            pr.append(avgedcla)
    dct = {'generation': gens, 'seed': seed_list,
           'prop_resampled': prop_resampled}
    dctcla = {'generation': gens, 'seed': seed_list, 'prop_resampled': pr}
    avgd_df = pd.DataFrame(dct)
    avgd_dfcla = pd.DataFrame(dctcla)

    fig, ax = plt.subplots()
    ax.set_xlabel('Generations')
    ax.set_ylabel('Resampled proprtion')
    sns.lineplot(avgd_df, x='generation', y='prop_resampled')
    sns.lineplot(avgd_dfcla, x='generation', y='prop_resampled')

    # plt.title('Resampled proportion of batches per generation')
    plt.savefig(os.path.join(save_dir, "to_resample.pdf"),
                dpi=400, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='ColoredMNIST/ColoredMNIST_debugging')
    parser.add_argument('--partition', type=str, default='test',
                        help="Evaluation partition, 'test' or 'valid'.")
    args = parser.parse_args()
    compare_name = ''
    dataset = args.dataset
    valid = args.partition
    save_dir = f'./replotted/comparisons/{dataset}_{valid}'
    res_dir = f'./results/{dataset}'
    if dataset == 'celeba':
        xlim = 8
        shade = 1
    elif dataset == 'ColoredMNIST':
        xlim = 40
        shade = 10
    else:
        xlim = 40
        shade = 5

    # #### NOMC Setting ####
    # agg_ar, over_ar, _ = read_csvs(res_dir, dataset, 'ARNOMC25', valid, False)
    # agg_no, over_no, _ = read_csvs(res_dir, dataset, 'NOMC25', valid, False)
    # arnomc(agg_ar, agg_no, save_dir, 'arnomc_nomc', xlim)
    # arnomcoveralls(over_ar, over_no, save_dir, 'arnomc_nomc', xlim)

    #### MC Setting ####
    agg_gar, over_gar, pop_gar = read_csvs(
        res_dir, dataset, 'ARGEN', valid, True)
    if dataset == 'ColoredMNIST':
        agg_sq, over_sq, pop_sq = read_csvs(
            res_dir, dataset, 'MC', valid, True)
        agg_ar, over_ar, pop_ar = read_csvs(
            res_dir, dataset, 'AR', valid, True)
    else:
        agg_sq, over_sq, pop_sq = read_csvs(
            res_dir, dataset, 'MC', valid, True)
        agg_ar, over_ar, pop_ar = read_csvs(
            res_dir, dataset, 'ARCLA', valid, True)

    armc(agg_ar, agg_sq, agg_gar, save_dir, 'argen_mc_ar', xlim, shade)
    aroveralls(over_ar, over_sq, over_gar,
               save_dir, 'argen_mc_ar', xlim, shade)
    arpop(pop_ar, pop_sq, pop_gar, save_dir, 'argen_mc_ar', xlim, shade)

    #### MC NOSEQ Setting ####
    # agg_sq, over_sq, pop_sq = read_csvs(res_dir, dataset, 'NOSEQ_MC', valid, True)
    # agg_ar, over_ar, pop_ar = read_csvs(res_dir, dataset, 'MC', valid, True)
    # arnomc(agg_ar, agg_sq, save_dir, 'arnomc_nomc')
    # arnomcoveralls(over_ar, over_sq, save_dir, 'arnomc_nomc')
    # rates(agg_ar, agg_sq, save_dir, compare_name)
    # armc(agg_ar, agg_sq, agg_gar, save_dir, 'argen_mc_ar', xlim)
    # aroveralls(over_ar, over_sq, over_gar, save_dir, 'argen_mc_ar', xlim)
    # arpop(pop_ar, pop_sq, pop_gar, save_dir, 'argen_mc_ar', xlim)
