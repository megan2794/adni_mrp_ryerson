from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from imblearn.over_sampling import SMOTE


def split(data, method, rs, k=5):

    X, y = data.drop('DX', axis=1), data['DX']
    data_dict = {}
    fold = 1

    # Upsample using SMOTE
    sm = SMOTE(sampling_strategy='auto', random_state=rs)
    X, y = sm.fit_resample(X, y)

    k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=rs)
    k_split = k_fold.split(X, y)

    if method == 'k-fold':
        for train_index, test_index in k_split:
            data_dict[str(fold)] = [X.iloc[train_index, :], y.iloc[train_index],
                                    X.iloc[test_index, :],  y.iloc[test_index]]
            fold = fold + 1
    else:
        data_dict[str(fold)] = [train_test_split(X, y, random_state=rs, test_size=0.33, shuffle=False)]

    return data_dict
