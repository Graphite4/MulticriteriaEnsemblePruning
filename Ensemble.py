
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score,  balanced_accuracy_score as bac
from imblearn.metrics import geometric_mean_score as g_mean
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from deap import base, creator, tools, algorithms

from random import randrange, randint

import numpy as np
import math


class MCE(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator_pool=None, no_bags=100, quality_metric="bac", diversity_measure="q", pruning_strategy="quality"):
        self._base_estimator_pool=base_estimator_pool
        self._no_bags = no_bags
        self._quality_metric = quality_metric
        self._diversity_measure = diversity_measure
        self._pruning_strategy = pruning_strategy

        self.ensemble_ = []
        self.classes_ = None

        self.X_ = None
        self.y_ = None

        self._X_train = None
        self._y_train = None
        self._X_valid = None
        self._y_valid = None

        self._y_predict = None

    @staticmethod
    def Q_statistic(classifiers_predict, y_true):
        def pairwise_Q_stat(cls1_res, cls2_result):
            N_0_0 = sum(cls1_res[cls2_result == 0] == 0)
            N_0_1 = sum(cls1_res[cls2_result == 1] == 0)
            N_1_0 = sum(cls1_res[cls2_result == 0] == 1)
            N_1_1 = sum(cls1_res[cls2_result == 1] == 1)
            return (N_1_1*N_0_0 - N_0_1*N_1_0) / (N_1_1*N_0_0 + N_0_1*N_1_0)

        results = [cls_pred == y_true for cls_pred in classifiers_predict]
        q_stat = 0
        for i in range(len(results)-1):
            for j in range(i+1,len(results)):
                q_stat += pairwise_Q_stat(results[i], results[j])
        return q_stat * 2 / (len(results) * (len(results)-1))



    @staticmethod
    def _evaluate(individual, y_predicts, y_true):
        classifiers_prediction = []
        for i in range(len(individual)):
            if individual[i] == 1:
                classifiers_prediction.append(y_predicts)
        predictions = np.array(classifiers_prediction)
        y_predict = MCE._majority_voting(predictions)
        qual = bac(y_true, y_predict)
        div = MCE.Q_statistic(predictions, y_true)
        return qual, div




    def _genetic_optimalisation(self):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

        IND_SIZE= len(self.ensemble_)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selNSGA2,  k=100)
        toolbox.register("evaluate", MCE._evaluate, y_predicts=self._y_predict, y_true=self.y_)



        result = algorithms.eaMuCommaLambda(toolbox.population(n=100), toolbox, 100, 100, 0.2, 0.1, 500)
        fitnesses = list(map(toolbox.evaluate, result))

        return result, fitnesses

    @staticmethod
    def _majority_voting(y_predict):
        acc_y_predict = np.empty((len(y_predict,)))
        for i in range(len(y_predict)):
            acc_y_predict = np.bincount(y_predict[:, i]).argmax()
        return acc_y_predict


    def _prune(self):
        pareto_set, fintesses = self._genetic_optimalisation()
        if self._pruning_strategy == 'quality':



    def fit(self, X, y):

        def subsample(X, y, ratio=1.0):
            n_sample = round(len(X) * ratio)
            sample_X = np.empty((n_sample, X.shape[1]))
            sample_y = np.empty((n_sample, y.shape[1]))
            for i in range(n_sample):
                index = randrange(len(X))
                sample_X[i, :] = X[index, :]
                sample_y[i, :] = y[index, :]
            return sample_X, sample_y

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        for train_index, test_index in sss.split(X, y):
            self._X_train, self._X_valid = X[train_index], X[test_index]
            self._y_train, self._y_valid = y[train_index], y[test_index]

        for e in self._base_estimator_pool:
            for i in range(self._no_bags):
                X_sample, y_sample = subsample(self._X_train, self._y_valid)
                new_e = clone(e)
                new_e.fit(X_sample, y_sample)
                self.ensemble_.append(new_e)

        self._y_predict = np.array([member_clf.predict(self._X_valid) for member_clf in self.ensemble_])







    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        return sum(prediction == y) / len(y)


    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        """Aposteriori probabilities."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Weight support before acumulation
        weighted_support = (
               self.ensemble_support_matrix(X) * self.weights_[:, np.newaxis, np.newaxis]
        )

        # Acumulate supports
        acumulated_weighted_support = np.sum(weighted_support, axis=0)
        return acumulated_weighted_support

    def predict(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        supports = self.predict_proba(X)
        prediction = np.argmax(supports, axis=1)

        return self.classes_[prediction]
