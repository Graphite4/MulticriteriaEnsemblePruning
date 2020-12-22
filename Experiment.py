from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score,  balanced_accuracy_score as bac
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from collections import Counter

import pandas as pd
import numpy as np
from Ensemble import MCE


file_list = [
             'segmentation.csv',
             'waveform.csv']
data_set = []
metrics = [bac]

for file in file_list:
    df = pd.read_csv(file)
    data_set.append(df)

def check_mean_composition(c):
    d = dict(c)
    for k, v in d.items():
        d[k] = v/10
    return d

def experiment(data_X, data_y, classifiers, pruning_strategy=None, **kwargs):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    pruning_strategy_list = ['quality', 'diversity', 'balanced', 'quality_single', 'diversity_single']
    if pruning_strategy is None:
        c = [Counter() for p in pruning_strategy_list]
        metrics_array = np.empty((len(pruning_strategy_list), 10, len(metrics)))
    else:
        c = Counter()
        metrics_array = np.empty((10, len(metrics)))


    i = 0
    for train_index, test_index in sss.split(data_X, data_y):
        for j in range(2):
            mce = MCE(base_estimator_pool=classifiers, pruning_strategy=pruning_strategy, **kwargs)
            mce.fit(data_X[train_index], data_y[train_index])
            if pruning_strategy is None:
                for inp, p in enumerate(pruning_strategy_list):
                    mce.set_ensemble(p)
                    c[inp] += mce.get_ensemble_composition()
                    y_predict = mce.predict(data_X[test_index])
                    for ind, it in enumerate(metrics):
                        metrics_array[inp, i, ind] = it(data_y[test_index], y_predict)
            else:
                c += mce.get_ensemble_composition()
                y_predict = mce.predict(data_X[test_index])
                for ind, it in enumerate(metrics):
                    metrics_array[i, ind] = it(data_y[test_index], y_predict)
            i +=1
            train_index, test_index = test_index, train_index
    if pruning_strategy is None:
        return np.mean(metrics_array, axis=1), [check_mean_composition(co) for co in c]
    else:
        return np.mean(metrics_array, axis=0), check_mean_composition(c)


def get_dataset(whole_ds, class_label):
    y = whole_ds[class_label].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    cols = list(whole_ds.columns)
    cols.remove(class_label)
    X = whole_ds[cols].values
    return X, y


def prepare_classifiers():
    ada = AdaBoostClassifier(random_state=42)
    r_forest = RandomForestClassifier(random_state=4)
    nb = GaussianNB()
    knn = KNeighborsClassifier()
    ann = MLPClassifier(random_state=666)
    # svc = LinearSVC(random_state=13)
    tree = DecisionTreeClassifier(random_state=7)
    return [ada, r_forest, nb, knn, ann, tree]

def conduct_experiments():
    classifiers = prepare_classifiers()
    pruning_strategy = ['quality','diversity','balanced', 'quality_single', 'doversity_single']
    results = pd.DataFrame(index=file_list, columns=pruning_strategy)

    for file_name, data in zip(file_list, data_set):
        ensemble_compositions = pd.DataFrame()
        X, y = get_dataset(data, "label")
        # for s in pruning_strategy:
        #     r, d = experiment(X, y, classifiers, s)
        #     results.at[file_name, s] = r
        #     ser = pd.Series(d)
        #     ser = ser.rename(s)
        #     ensemble_compositions = ensemble_compositions.append(ser)
        r, d = experiment(X, y, classifiers)
        # results.at[file_name, 'quality'] = r[0]
        # results.at[file_name, 'diversity'] = r[1]
        # results.at[file_name, 'balanced'] = r[2]
        # results.at[file_name, 'quality_single'] = r[3]
        # results.at[file_name, 'diversity_single'] = r[4]
        for i, s in enumerate(pruning_strategy):
            results.at[file_name, s] = r[i]
            ser = pd.Series(d[i])
            ser = ser.rename(s)
            ensemble_compositions = ensemble_compositions.append(ser)
        # r2, d2 = experiment(X, y, classifiers, 'quality', diversity_measure=None)
        # results.at[file_name, 'quality_single'] = r2
        # ser = pd.Series(d2)
        # ser = ser.rename('quality_single')
        # ensemble_compositions = ensemble_compositions.append(ser)
        # r3, d3 = experiment(X, y, classifiers, 'diversity', quality_metric=None)
        # results.at[file_name, 'diversity_single'] = r3
        # ser = pd.Series(d3)
        # ser = ser.rename('diversity_single')
        # ensemble_compositions = ensemble_compositions.append(ser)

        results.to_csv(path_or_buf=file_name.split('.')[0] + "_Wyniki4.csv")
        ensemble_compositions.to_csv(path_or_buf=file_name.split('.')[0] + "_Sklad4.csv")


if __name__ == "__main__":
    conduct_experiments()



        













































