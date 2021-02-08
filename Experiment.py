from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score,precision_score as precision, recall_score as recall,  balanced_accuracy_score as bac
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from collections import Counter

import pandas as pd
import numpy as np
from DatasetsCollection import load
from Ensemble import MCE


# file_list = ['pima.csv',
#              'vowel0.csv',
#              'segment0.csv']
# file_list = [
file_list = ["abalone19", "abalone9-18", "ecoli-0-1-3-7_vs_2-6", "glass-0-1-6_vs_2", "glass-0-1-6_vs_5", "glass2",
             "glass4", "glass5", "page-blocks-1-3_vs_4", "yeast-0-5-6-7-9_vs_4", "yeast-1-2-8-9_vs_7",
             "yeast-1-4-5-8_vs_7", "yeast-1_vs_7", "yeast-2_vs_4", "yeast-2_vs_8", "yeast4", "yeast5", "yeast6",
             "cleveland-0_vs_4", "ecoli-0-1-4-7_vs_2-3-5-6", "ecoli-0-1_vs_2-3-5", "ecoli-0-2-6-7_vs_3-5",
             "ecoli-0-6-7_vs_3-5", "ecoli-0-6-7_vs_5", "glass-0-1-4-6_vs_2", "glass-0-1-5_vs_2",
             "yeast-0-2-5-6_vs_3-7-8-9", "yeast-0-3-5-9_vs_7-8", "abalone-17_vs_7-8-9-10", "abalone-19_vs_10-11-12-13",
             "abalone-20_vs_8-9-10", "abalone-21_vs_8", "flare-F", "kddcup-buffer_overflow_vs_back",
             "kddcup-rootkit-imap_vs_back", "kr-vs-k-zero_vs_eight", "poker-8-9_vs_5", "poker-8-9_vs_6", "poker-8_vs_6",
             "poker-9_vs_7", "winequality-red-3_vs_5", "winequality-red-4", "winequality-red-8_vs_6-7",
             "winequality-red-8_vs_6", "winequality-white-3-9_vs_5", "winequality-white-3_vs_7",
             "winequality-white-9_vs_4", "zoo-3", "ecoli1", "ecoli2", "ecoli3", "glass0", "glass1", "haberman",
             "page-blocks0", "pima", "vehicle1", "vehicle3", "yeast1", "yeast3"
]
data_set = []
metrics = [bac, precision, recall]


def check_mean_composition(c):
    d = dict(c)
    for k, v in d.items():
        d[k] = v/10
    return d

# def experiment(data_X, data_y, classifiers, pruning_strategy=None, **kwargs):


def experiment(data, classifiers, pruning_strategy=None, **kwargs):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    pruning_strategy_list = ['quality', 'diversity', 'balanced', 'quality_single', 'precision_single', 'recall_single']
    if pruning_strategy is None:
        c = [Counter() for p in pruning_strategy_list]
        metrics_array = np.empty((len(pruning_strategy_list), 10, len(metrics)))
    else:
        c = Counter()
        metrics_array = np.empty((10, len(metrics)))


    # i = 0
    # for train_index, test_index in sss.split(data_X, data_y):
    #     for j in range(2):
    #         mce = MCE(base_estimator_pool=classifiers, pruning_strategy=pruning_strategy, **kwargs)
    #         mce.fit(data_X[train_index], data_y[train_index])
    #         if pruning_strategy is None:
    #             for inp, p in enumerate(pruning_strategy_list):
    #                 mce.set_ensemble(p)
    #                 c[inp] += mce.get_ensemble_composition()
    #                 y_predict = mce.predict(data_X[test_index])
    #                 for ind, it in enumerate(metrics):
    #                     metrics_array[inp, i, ind] = it(data_y[test_index], y_predict)
    #         else:
    #             c += mce.get_ensemble_composition()
    #             y_predict = mce.predict(data_X[test_index])
    #             for ind, it in enumerate(metrics):
    #                 metrics_array[i, ind] = it(data_y[test_index], y_predict)
    #         i +=1
    #         train_index, test_index = test_index, train_index
    i = 0
    for fold in data:
        mce = MCE(base_estimator_pool=classifiers, pruning_strategy=pruning_strategy, **kwargs)
        mce.fit(fold[0][0], fold[0][1])
        if pruning_strategy is None:
            for inp, p in enumerate(pruning_strategy_list):
                mce.set_ensemble(p)
                c[inp] += mce.get_ensemble_composition()
                y_predict = mce.predict(fold[1][0])
                for ind, it in enumerate(metrics):
                    metrics_array[inp, i, ind] = it(fold[1][1], y_predict)
        else:
            c += mce.get_ensemble_composition()
            y_predict = mce.predict(fold[1][0])
            for ind, it in enumerate(metrics):
                metrics_array[i, ind] = it(fold[1][1], y_predict)
        i += 1

    if pruning_strategy is None:
        return np.mean(metrics_array, axis=1), [check_mean_composition(co) for co in c]
    else:
        return np.mean(metrics_array, axis=0), check_mean_composition(c)


def prepare_dataset(whole_ds, class_label):
    try:
        whole_ds.drop(columns='Unnamed: 0', inplace=True)
    except:
        pass
    y = whole_ds[class_label].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    cols = list(whole_ds.columns)
    cols.remove(class_label)
    X = whole_ds[cols].values
    return X, y

def get_datasets():

    # for file in file_list:
    #     df = pd.read_csv(file)
    #     data_set.append(df)

    for file in file_list:
        data_set.append(load(file))

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
    get_datasets()
    pruning_strategy = ['precision','recall','balanced', 'balanced_accuracy_single', 'precision_single', 'recall_single']

    results = [pd.DataFrame(index=file_list, columns=pruning_strategy) for m in metrics]
    for j in [20, 50, 100]:
        for file_name, data in zip(file_list, data_set):
            ensemble_compositions = pd.DataFrame()
            # X, y = prepare_dataset(data, "label")
            r, d = experiment(data, classifiers, no_bags=j)
            for i_m, m in enumerate(metrics):
                for i, s in enumerate(pruning_strategy):
                    results[i_m].at[file_name, s] = r[i, i_m]

            for i, s in enumerate(pruning_strategy):
                ser = pd.Series(d[i])
                ser = ser.rename(s)
                ensemble_compositions = ensemble_compositions.append(ser)

                ensemble_compositions.to_csv(path_or_buf="wyniki_keel_sklady/" +file_name.split('.')[0] + "_sklad_random_oversampling_NEW" + str(j) + "_bags.csv")

            for i_m, m in enumerate(metrics):
                results[i_m].to_csv("wyniki_keel/wyniki_random_oversampling_NEW"+str(j)+"_bags_"+m.__name__)


if __name__ == "__main__":
    conduct_experiments()



        













































