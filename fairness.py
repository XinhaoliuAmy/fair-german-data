import utils
import numpy as np
import fairlearn.metrics as fairmets
from sklearn.metrics import accuracy_score
import pandas as pd

metrics_dict = {
    'Accuracy': accuracy_score,
    'Selection rate': fairmets.selection_rate,
    'Count': fairmets.count,
    'tpr': fairmets.true_positive_rate,
    'tnr': fairmets.true_negative_rate,
    'fpr': fairmets.false_positive_rate,
    'fnr': fairmets.false_negative_rate
}


def eval_classifier(labels, preds, sensitive):
    mf2 = fairmets.MetricFrame(
        metrics=metrics_dict, y_true=labels, y_pred=preds, sensitive_features=sensitive)
    by_group = mf2.by_group
    mf3 = pd.DataFrame({'difference': mf2.difference(),
                        # 'ratio': mf2.ratio(),
                        'group_min': mf2.group_min(),
                        'group_max': mf2.group_max(),
                        'group_0': by_group.iloc[0],
                        'group_1': by_group.iloc[1]}).T

    mf3_data = mf3.to_numpy().flatten()
    label_list = []
    for rown in mf3.index.to_list():
        for cname in mf3.columns.to_list():
            label_list.append(f"{rown}_{cname}")

    overall_labels = mf2.overall.index.to_list()
    overall = mf2.overall.to_numpy().flatten()

    return [label_list, mf3_data, overall_labels, overall]
