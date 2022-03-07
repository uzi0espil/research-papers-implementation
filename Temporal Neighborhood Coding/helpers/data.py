import pandas as pd
import numpy as np


def build(X, y, users):
    X_groups= []
    y_groups = []
    min_len = np.inf
    for user, group in users.groupby("user_id"):
        features_group = X.iloc[group.index]
        labels_group = y.iloc[group.index]
        X_groups.append(features_group)
        y_groups.append(labels_group)
        min_len = min_len if min_len < features_group.shape[0] else features_group.shape[0]
    X_stack = [x[:min_len] for x in X_groups]
    y_stack = [y[:min_len] for y in y_groups]
    return X_stack, y_stack