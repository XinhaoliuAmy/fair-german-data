import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import utils
import os
import seaborn as sns
from matplotlib.lines import Line2D


plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': False,
    'pgf.rcfonts': False,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'axes.titlepad': 5,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.linewidth': 2,
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
blue = 'tab:blue'
purple = 'tab:purple'
lw = 2
alph = .2


def remove_dup_runs(df):
    df = df.drop_duplicates(keep='last', subset=['generation'], ignore_index=True)
    return df

def read_csvs(res_dir, dataset, exp_id, fname):
    last_seed = 9 #utils.get_last_seed(res_dir, f"{dataset}_{exp_id}_")
    frames = []
    for seed in range(last_seed + 1):
        res_file = os.path.join(res_dir, f"{dataset}_{exp_id}_{seed}", fname)
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

def init_save_dir(save_dir, overwrite):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:  # figures already exist, check before overwriting
        if overwrite == 0:
            print(f"Saving directory {save_dir} is not empty, stopping.")
            return False
    return True

def nomc_aggregated(frames, colors, linestyles, names, save_dir, compare_name, xlim=40):
    for frame in frames:
        tprfpr_diffs = frame[['difference_tpr', 'difference_fpr']]
        frame['eodds_diff'] = tprfpr_diffs.max(axis=1)

    ys = ['difference_Accuracy', 'difference_Selection rate', 'eodds_diff']
    saves = [f'{save_dir}/{compare_name}_acc_diff.pdf', f'{save_dir}/{compare_name}_dp.pdf', f'{save_dir}/{compare_name}_eodds.pdf']
    labels = ['Accuracy Difference', 'DP Difference', 'EOdds Difference']
    for y, save, lab in zip(ys, saves, labels):
        for frame, c, ls in zip(frames, colors, linestyles):
            sns.lineplot(frame, x='generation', y=y, color=c, legend=False, linestyle=ls, linewidth=lw)
        plt.ylabel(lab)
        plt.xlabel('Generations')
        plt.xlim((0, xlim))
        legend_elems = [Line2D([0], [0], color=c, lw=4, linestyle=ls) for c, ls in zip(colors, linestyles)]
        plt.legend(legend_elems, names)
        plt.savefig(save, dpi=400, bbox_inches='tight')
        plt.close()

def nomc_overalls(frames, colors, linestlyes, names, save_dir, compare_name, xlim=40):
    for frame, c, ls in zip(frames, colors, linestyles):
        sns.lineplot(frame, x='generation', y='Accuracy', color=c, legend=False, linestyle=ls, linewidth=lw)
    plt.ylabel('Accuracy')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    plt.ylim((.9, .95))
    legend_elems = [Line2D([0], [0], color=c, lw=4, linestyle=ls) for c, ls in zip(colors, linestlyes)]
    plt.legend(legend_elems, names)
    plt.savefig(f'{save_dir}/{compare_name}_acc.pdf', dpi=400, bbox_inches='tight')
    plt.close()

def mc_aggregated(half, fancy, claf, genf, save_dir, compare_name, xlim=40, shade=15):
    sns.lineplot(claf, x='generation', y='difference_Selection rate', color=green, legend=False, linestyle='dotted', linewidth=lw)
    sns.lineplot(half, x='generation', y='difference_Selection rate', color=red, legend=False, linewidth=lw)
    sns.lineplot(fancy, x='generation', y='difference_Selection rate', color=blue, legend=False, linestyle='dashed', linewidth=lw)
    sns.lineplot(genf, x='generation', y='difference_Selection rate', color=purple, legend=False, linestyle='dashdot', linewidth=lw)
    plt.ylabel('Demographic Parity Diff')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dotted'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color=blue, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=purple, lw=4, linestyle='dashdot')]
    plt.legend(legend_elems, ['CLAFancy', 'Half ', 'Fancy', 'GENFancy'])
    # plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=alph)
    plt.savefig(f'{save_dir}/{compare_name}_dp.pdf', dpi=400, bbox_inches='tight')
    plt.close()

    sns.lineplot(claf, x='generation', y='difference_Accuracy', color=green, legend=False, linestyle='dotted', linewidth=lw)
    sns.lineplot(half, x='generation', y='difference_Accuracy', color=red, legend=False, linewidth=lw)
    sns.lineplot(fancy, x='generation', y='difference_Accuracy', color=blue, legend=False, linestyle='dashed', linewidth=lw)
    sns.lineplot(genf, x='generation', y='difference_Accuracy', color=purple, legend=False, linestyle='dashdot', linewidth=lw)
    plt.ylabel('Accuracy Diff')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dotted'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color=blue, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=purple, lw=4, linestyle='dashdot')]
    plt.legend(legend_elems, ['CLAFancy', 'Half ', 'Fancy', 'GENFancy'])
    # plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=alph)
    plt.savefig(f'{save_dir}/{compare_name}_acc_diff.pdf', dpi=400, bbox_inches='tight')
    plt.close()

    tprfpr_diffs = half[['difference_tpr', 'difference_fpr']]
    half['eodds_diff'] = tprfpr_diffs.max(axis=1)
    tprfpr_diffs = claf[['difference_tpr', 'difference_fpr']]
    claf['eodds_diff'] = tprfpr_diffs.max(axis=1)
    tprfpr_diffs = genf[['difference_tpr', 'difference_fpr']]
    genf['eodds_diff'] = tprfpr_diffs.max(axis=1)
    tprfpr_diffs = fancy[['difference_tpr', 'difference_fpr']]
    fancy['eodds_diff'] = tprfpr_diffs.max(axis=1)
    sns.lineplot(claf, x='generation', y='eodds_diff', color=green, legend=False, linestyle='dotted', linewidth=lw)
    sns.lineplot(half, x='generation', y='eodds_diff', color=red, legend=False, linewidth=lw)
    sns.lineplot(fancy, x='generation', y='eodds_diff', color=blue, legend=False, linestyle='dashed', linewidth=lw)
    sns.lineplot(genf, x='generation', y='eodds_diff', color=purple, legend=False, linestyle='dashdot', linewidth=lw)
    plt.ylabel('Equalized Odds Diff')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=green, lw=4, linestyle='dotted'),
                    Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color=blue, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=purple, lw=4, linestyle='dashdot')]
    plt.legend(legend_elems, ['CLAFancy', 'Half ', 'Fancy', 'GENFancy'])
    # plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=alph)
    plt.savefig(f'{save_dir}/{compare_name}_eodds.pdf', dpi=400, bbox_inches='tight')
    plt.close()

def mc_overalls(half, fancy, claf, genf, save_dir, compare_name, xlim=40, shade=15):
    sns.lineplot(half, x='generation', y='Accuracy', color=red, legend=False, linewidth=lw)
    sns.lineplot(fancy, x='generation', y='Accuracy', color=blue, legend=False, linestyle='dashed', linewidth=lw)
    sns.lineplot(claf, x='generation', y='Accuracy', color=green, legend=False, linestyle='dotted', linewidth=lw)
    sns.lineplot(genf, x='generation', y='Accuracy', color=purple, legend=False, linestyle='dashdot', linewidth=lw)
    plt.ylabel('Accuracy')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color=blue, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=green, lw=4, linestyle='dotted'),
                    Line2D([0], [0], color=purple, lw=4, linestyle='dashdot')]
    plt.legend(legend_elems, ['Half', 'Fancy', 'CLAFancy', 'GENFancy'])
    # plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=alph)
    plt.savefig(f'{save_dir}/{compare_name}_acc.pdf', dpi=400, bbox_inches='tight')
    plt.close()

    sns.lineplot(half, x='generation', y='Selection rate', color=red, legend=False, linewidth=lw)
    sns.lineplot(fancy, x='generation', y='Selection rate', color=blue, legend=False, linestyle='dashed', linewidth=lw)
    sns.lineplot(claf, x='generation', y='Selection rate', color=green, legend=False, linestyle='dotted', linewidth=lw)
    sns.lineplot(genf, x='generation', y='Selection rate', color=purple, legend=False, linestyle='dashdot', linewidth=lw)
    plt.ylabel('Selection Rate')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color=blue, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=green, lw=4, linestyle='dotted'),
                    Line2D([0], [0], color=purple, lw=4, linestyle='dashdot')]
    plt.legend(legend_elems, ['Half', 'Fancy', 'CLAFancy', 'GENFancy'])
    # plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=alph)
    plt.savefig(f'{save_dir}/{compare_name}_sel_rate.pdf', dpi=400, bbox_inches='tight')
    plt.close()
    
def mc_pop_stats(half, fancy, claf, genf, save_dir, compare_name, xlim=40, shade=15):
    sns.lineplot(half, x='generation', y='label_bal', color=red, legend=False, linewidth=lw)
    sns.lineplot(fancy, x='generation', y='label_bal', color=blue, legend=False, linestyle='dashed', linewidth=lw)
    sns.lineplot(claf, x='generation', y='label_bal', color=green, legend=False, linestyle='dotted', linewidth=lw)
    sns.lineplot(genf, x='generation', y='label_bal', color=purple, legend=False, linestyle='dashdot', linewidth=lw)
    plt.ylabel('Beneficial Class')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color=blue, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=green, lw=4, linestyle='dotted'),
                    Line2D([0], [0], color=purple, lw=4, linestyle='dashdot')]
    plt.legend(legend_elems, ['Half', 'Fancy', 'CLAFancy', 'GENFancy'])
    # plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=alph)
    plt.savefig(f'{save_dir}/{compare_name}_label_bal.pdf', dpi=400, bbox_inches='tight')
    plt.close()

    sns.lineplot(half, x='generation', y='color_bal', color=red, legend=False, linewidth=lw)
    sns.lineplot(fancy, x='generation', y='color_bal', color=blue, legend=False, linestyle='dashed', linewidth=lw)
    sns.lineplot(claf, x='generation', y='color_bal', color=green, legend=False, linestyle='dotted', linewidth=lw)
    sns.lineplot(genf, x='generation', y='color_bal', color=purple, legend=False, linestyle='dashdot', linewidth=lw)
    plt.ylabel('Minoritized Group')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color=blue, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=green, lw=4, linestyle='dotted'),
                    Line2D([0], [0], color=purple, lw=4, linestyle='dashdot')]
    plt.legend(legend_elems, ['Half', 'Fancy', 'CLAFancy', 'GENFancy'])
    # plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=alph)
    plt.savefig(f'{save_dir}/{compare_name}_color_bal.pdf', dpi=400, bbox_inches='tight')
    plt.close()

def kl_plots(half, fancy, claf, genf, save_dir, compare_name, model, xlim=40, shade=15):
    sns.lineplot(half, x='generation', y='kl_div', color=red, legend=False, linewidth=lw)
    sns.lineplot(fancy, x='generation', y='kl_div', color=blue, legend=False, linestyle='dashed', linewidth=lw)
    sns.lineplot(claf, x='generation', y='kl_div', color=green, legend=False, linestyle='dotted', linewidth=lw)
    sns.lineplot(genf, x='generation', y='kl_div', color=purple, legend=False, linestyle='dashdot', linewidth=lw)
    plt.ylabel('KL-Divergence')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=red, lw=4),
                    Line2D([0], [0], color=blue, lw=4, linestyle='dashed'),
                    Line2D([0], [0], color=green, lw=4, linestyle='dotted'),
                    Line2D([0], [0], color=purple, lw=4, linestyle='dashdot')]
    plt.legend(legend_elems, ['Half', 'Fancy', 'CLAFancy', 'GENFancy'])
    # plt.axvspan(xmin=shade, xmax=xlim, color='gray', alpha=alph)
    plt.savefig(f'{save_dir}/{compare_name}_{model}_kl_div.pdf', dpi=400, bbox_inches='tight')
    plt.close()

def nomc_kl_plots(frames, colors, linestyles, names, save_dir, compare_name, model, xlim=40, shade=15):
    for frame, c, ls in zip(frames, colors, linestyles):
        sns.lineplot(frame, x='generation', y='kl_div', color=c, legend=False, linestyle=ls, linewidth=lw)
    plt.ylabel('KL-Divergence')
    plt.xlabel('Generations')
    plt.xlim((0, xlim))
    legend_elems = [Line2D([0], [0], color=c, lw=4, linestyle=ls) for c, ls in zip(colors, linestyles)]
    plt.legend(legend_elems, names)
    plt.savefig(f'{save_dir}/{compare_name}_{model}_kl_div.pdf', dpi=400, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    nomc_comp_list = ['50', '50_F', '50_ARF']
    mc_comp_list = ['MC_50', 'MC_50_F', 'MC_50_CLAF', 'MC_50_GENF'] 
    dataset = 'ColoredMNIST'
    valid='test'
    save_dir = f'./fresh_figs/{dataset}_{valid}_HALFS'
    init_save_dir(save_dir, overwrite=0)
    res_dir = f'./results/{dataset}'

    # NOMC
    fancy_over = read_csvs(res_dir, dataset, '50_F', f'cla_{valid}_overall.csv')
    arfancy_over = read_csvs(res_dir, dataset, '50_ARF', f'cla_{valid}_overall.csv')
    half_over = read_csvs(res_dir, dataset, '50', f'cla_{valid}_overall.csv')
    full_over = read_csvs(res_dir, dataset, 'MC', f'cla_{valid}_overall.csv')
    frames = [full_over, half_over, fancy_over, arfancy_over]
    colors = [red, 'tab:orange', blue, green]
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    names = ['100% Synth', '50% Synth', '50% Synth + Fancy', '50% Synth + Fancy + Cla-STAR']
    nomc_overalls(frames, colors, linestyles, names, save_dir, 'nomc')

    fancy_agg = read_csvs(res_dir, dataset, '50_F', f'cla_{valid}_aggregated.csv')
    arfancy_agg = read_csvs(res_dir, dataset, '50_ARF', f'cla_{valid}_aggregated.csv')
    half_agg = read_csvs(res_dir, dataset, '50', f'cla_{valid}_aggregated.csv')
    # full_agg = read_csvs(res_dir, dataset, 'MC', f'cla_{valid}_aggregated.csv')
    frames = [half_agg, fancy_agg, arfancy_agg]
    colors = ['tab:orange', blue, green]
    linestyles = ['solid', 'dotted', 'dashdot']
    names = ['50% Synth', '50% Synth + Fancy', '50% Synth + Fancy + Cla-STAR']
    nomc_aggregated(frames, colors, linestyles, names, save_dir, 'nomc')

    # KL Stuff
    for model in ['cla']:
        fancy_over = read_csvs(res_dir, dataset, '50_F', f'{model}_strata_kl.csv')
        arfancy_over = read_csvs(res_dir, dataset, '50_ARF', f'{model}_strata_kl.csv')
        half_over = read_csvs(res_dir, dataset, '50', f'{model}_strata_kl.csv')
        frames = [half_over, fancy_over, arfancy_over]
        colors = ['tab:orange', blue, green]
        linestyles = ['solid', 'dotted', 'dashdot']
        names = ['50% Synth', '50% Synth + Fancy', '50% Synth + Fancy + Cla-STAR']
        nomc_kl_plots(frames, colors, linestyles, names, save_dir, 'nomc', model)

    # MC Stuff
    # mc_over = read_csvs(res_dir, dataset, 'MC', f'cla_{valid}_overall.csv')
    # half_over = read_csvs(res_dir, dataset, 'MC_50', f'cla_{valid}_overall.csv')
    # fancy_over = read_csvs(res_dir, dataset, 'MC_50_F', f'cla_{valid}_overall.csv')
    # # half_over = read_csvs(res_dir, dataset, 'MC_50_CLA', f'cla_{valid}_overall.csv')
    # # fancy_over = read_csvs(res_dir, dataset, 'MC_50_GEN', f'cla_{valid}_overall.csv')
    # claf_over = read_csvs(res_dir, dataset, 'MC_50_CLAF', f'cla_{valid}_overall.csv')
    # genf_over = read_csvs(res_dir, dataset, 'MC_50_GENF', f'cla_{valid}_overall.csv')
    # mc_overalls(half_over, fancy_over, claf_over, genf_over, save_dir, 'mc')

    # mc_agg = read_csvs(res_dir, dataset, 'MC', f'cla_{valid}_aggregated.csv')
    # half_agg = read_csvs(res_dir, dataset, 'MC_50', f'cla_{valid}_aggregated.csv')
    # fancy_agg = read_csvs(res_dir, dataset, 'MC_50_F', f'cla_{valid}_aggregated.csv')
    # # half_agg = read_csvs(res_dir, dataset, 'MC_50_CLA', f'cla_{valid}_aggregated.csv')
    # # fancy_agg = read_csvs(res_dir, dataset, 'MC_50_GEN', f'cla_{valid}_aggregated.csv')
    # claf_agg = read_csvs(res_dir, dataset, 'MC_50_CLAF', f'cla_{valid}_aggregated.csv')
    # genf_agg = read_csvs(res_dir, dataset, 'MC_50_GENF', f'cla_{valid}_aggregated.csv')
    # mc_aggregated(half_agg, fancy_agg, claf_agg, genf_agg, save_dir, 'mc')

    # # KL Stuff
    # for model in ['gen', 'cla']:
    #     mc_agg = read_csvs(res_dir, dataset, 'KL_MC', f'{model}_strata_kl.csv')
    #     half_agg = read_csvs(res_dir, dataset, 'MC_50', f'{model}_strata_kl.csv')
    #     fancy_agg = read_csvs(res_dir, dataset, 'MC_50_F', f'{model}_strata_kl.csv')
    #     claf_agg = read_csvs(res_dir, dataset, 'MC_50_CLAF', f'{model}_strata_kl.csv')
    #     genf_agg = read_csvs(res_dir, dataset, 'MC_50_GENF', f'{model}_strata_kl.csv')
    #     kl_plots(half_agg, fancy_agg, claf_agg, genf_agg, save_dir, 'mc', model)

    # # Population Stats for MC generators
    # mc_agg = read_csvs(res_dir, dataset, 'MC', 'gen_pop_stats.csv')
    # half_agg = read_csvs(res_dir, dataset, 'MC_50', 'gen_pop_stats.csv')
    # fancy_agg = read_csvs(res_dir, dataset, 'MC_50_F', 'gen_pop_stats.csv')
    # claf_agg = read_csvs(res_dir, dataset, 'MC_50_CLAF', 'gen_pop_stats.csv')
    # genf_agg = read_csvs(res_dir, dataset, 'MC_50_GENF', 'gen_pop_stats.csv')
    # mc_pop_stats(half_agg, fancy_agg, claf_agg, genf_agg, save_dir, 'mc')