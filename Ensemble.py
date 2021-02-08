
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score,  balanced_accuracy_score as bac
from deap import base, creator, tools, algorithms
from operator import itemgetter
from random import randrange, randint, sample
from collections import Counter
from platypus import nondominated, Problem, Real, Integer, Binary, unique
from platypus.algorithms import NSGAII
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from RandomSubspaceClassifierWrapper import RandomSubspaceClassifierWrapper

import numpy as np
import math
import functools


class MCE(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator_pool=None, no_bags=100, quality_metric="bac", diversity_measure="q", pruning_strategy="quality"):
        self._base_estimator_pool = base_estimator_pool
        self._no_bags = no_bags
        self._quality_metric = quality_metric
        self._diversity_measure = diversity_measure
        self._pruning_strategy = pruning_strategy

        self.ensemble_ = []
        self._ensemble_indices = []
        self.classes_ = None

        self.X_ = None
        self.y_ = None

        self._X_train = None
        self._y_train = None
        self._X_valid = None
        self._y_valid = None

        self._y_predict = None
        self._pairwise_diversity_stats = None

    def _get_pairwise_Q_stat(self):
        # def pairwise_Q_stat(cls1_res, cls2_result):
        #     N_0_0 = sum(cls1_res[cls2_result == 0] == 0)
        #     N_0_1 = sum(cls1_res[cls2_result == 1] == 0)
        #     N_1_0 = sum(cls1_res[cls2_result == 0] == 1)
        #     N_1_1 = sum(cls1_res[cls2_result == 1] == 1)
        #     return (N_1_1*N_0_0 - N_0_1*N_1_0) / (N_1_1*N_0_0 + N_0_1*N_1_0)
        def pairwise_Q_stat(cls1_res, cls2_res):
            N_0_0 = len(np.where((cls1_res == 0) & (cls2_res == 0))[0])
            N_0_1 = len(np.where((cls1_res == 0) & (cls2_res == 1))[0])
            N_1_0 = len(np.where((cls1_res == 1) & (cls2_res == 0))[0])
            N_1_1 = len(cls1_res) - (N_0_0+N_0_1+N_1_0)
            if (N_1_1*N_0_0 + N_0_1*N_1_0) == 0:
                return 1
            return (N_1_1*N_0_0 - N_0_1*N_1_0) / (N_1_1*N_0_0 + N_0_1*N_1_0)
        results = [cls_pred == self._y_valid for cls_pred in self._y_predict]

        for i in range(len(self.ensemble_)):
            for j in range(i+1, len(self.ensemble_)):
                q_stat = pairwise_Q_stat(results[i], results[j])
                self._pairwise_diversity_stats[i, j] = q_stat
                self._pairwise_diversity_stats[j, i] = q_stat


    @staticmethod
    def Q_statistic(individual, pairwise_q_stat):
        classifiers = np.where(np.array(individual) == 1)[0]
        if len(classifiers) == 1:
            return 1
        if len(classifiers) == 0:
            return 100
        q_stat = 0
        for i in range(len(classifiers)):
            for j in range(i+1, len(classifiers)):
                q_stat += pairwise_q_stat[i, j]
        return q_stat * 2 / (len(classifiers) * (len(classifiers)-1))

    @staticmethod
    def get_group(code, full_list):
        # group_list = []
        # for i in range(len(code)):
        #     if code[i] == 1:
        #         group_list.append(full_list[i])
        # group = np.array(group_list)
        if isinstance(full_list, list):
            return list(np.array(full_list)[np.where(np.array(code) == 1)[0]])
        else:
            return full_list[np.where(np.array(code) == 1)[0]]

    @staticmethod
    def _evaluate(individual, y_predicts, y_true, pairwise_div_stat):
        predictions = MCE.get_group(individual, y_predicts)
        if predictions.size > 0:
            y_predict = MCE._majority_voting(predictions)
            qual = bac(y_true, y_predict)
        else:
            qual = 0
        div = MCE.Q_statistic(individual, pairwise_div_stat)
        return qual, div

    @staticmethod
    def _evaluate_imbalance(individual, y_predicts, y_true):
        predictions = MCE.get_group(individual, y_predicts)
        if predictions.size > 0:
            y_predict = MCE._majority_voting(predictions)
            qual1 = precision_score(y_true.astype("int8"), y_predict.astype("int8"))
            qual2 = recall_score(y_true.astype("int8"), y_predict.astype("int8"))
        else:
            qual1 = 0
            qual2 = 0
        return qual1, qual2

    @staticmethod
    def _evaluate_p(individual, y_predicts, y_true):
        predictions = MCE.get_group(individual, y_predicts)
        y_predict = MCE._majority_voting(predictions)
        qual = precision_score(y_true.astype("int8"), y_predict.astype("int8"))
        return (qual,)

    @staticmethod
    def _evaluate_r(individual, y_predicts, y_true):
        predictions = MCE.get_group(individual, y_predicts)
        y_predict = MCE._majority_voting(predictions)
        try:
            qual = recall_score(y_true.astype("int8"), y_predict.astype("int8"))
        except:
            print('UPS')
        return (qual,)

    @staticmethod
    def _evaluate_q(individual, y_predicts, y_true):
        predictions = MCE.get_group(individual, y_predicts)
        y_predict = MCE._majority_voting(predictions)
        qual = bac(y_true, y_predict)
        return (qual,)

    @staticmethod
    def _evaluate_d(individual, pairwise_div_stat):
        div = MCE.Q_statistic(individual, pairwise_div_stat)
        return (div,)

    def _genetic_optimalisation(self, optimalisation_type='multi'):
        if optimalisation_type == 'diversity_single':
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0,))
        elif optimalisation_type == 'quality_single':
            creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
        elif optimalisation_type == 'precision_single':
            creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
        elif optimalisation_type == 'recall_single':
            creator.create("FitnessMulti", base.Fitness, weights=(1.0,))
        else:
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMulti)

        IND_SIZE= len(self.ensemble_)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        if optimalisation_type == 'multi':
            toolbox.register("select", tools.selNSGA2)
            toolbox.register("evaluate", MCE._evaluate, y_predicts=self._y_predict, y_true=self._y_valid, pairwise_div_stat=self._pairwise_diversity_stats)
        else:
            toolbox.register("select", tools.selTournament, tournsize=50)
            if optimalisation_type == 'quality_single':
                toolbox.register("evaluate", MCE._evaluate_q, y_predicts=self._y_predict, y_true=self._y_valid)
            elif optimalisation_type == 'precision_single':
                toolbox.register("evaluate", MCE._evaluate_p, y_predicts=self._y_predict, y_true=self._y_valid)
            elif optimalisation_type == 'recall_single':
                toolbox.register("evaluate", MCE._evaluate_r, y_predicts=self._y_predict, y_true=self._y_valid)
            else:
                toolbox.register("evaluate", MCE._evaluate_d, pairwise_div_stat=self._pairwise_diversity_stats)


        result = algorithms.eaMuCommaLambda(toolbox.population(n=100), toolbox, 100, 100, 0.2, 0.1, 500)[0]
        fitnesses = list(map(toolbox.evaluate, result))

        return result, fitnesses

    @staticmethod
    def _majority_voting(y_predict):
        try:
            acc_y_predict = np.empty((y_predict.shape[1],))
            y_predict = y_predict.astype(int)
            for i in range(y_predict.shape[1]):
                acc_y_predict[i] = np.bincount(y_predict[:, i]).argmax()
        except:
            print('UPS')
        return acc_y_predict

    def _prune(self):
        problem = Problem(len(self.ensemble_), 2)
        problem.types[:] = Integer(0, 1)
        problem.directions[0] = Problem.MAXIMIZE
        problem.directions[1] = Problem.MAXIMIZE
        problem.function = functools.partial(MCE._evaluate_imbalance, y_predicts=self._y_predict, y_true=self._y_valid)

        algorithm = NSGAII(problem)
        algorithm.run(10000)

        solutions = unique(nondominated(algorithm.result))
        objectives = [sol.objectives for sol in solutions]

        def extract_variables(variables):
            extracted = [v[0] for v in variables]
            return extracted

        self._ensemble_quality = self.get_group(extract_variables(solutions[objectives.index(max(objectives, key=itemgetter(0)))].variables), self.ensemble_)
        self._ensemble_diversity = self.get_group(extract_variables(solutions[objectives.index(max(objectives, key=itemgetter(1)))].variables), self.ensemble_)
        self._ensemble_balanced = self.get_group(extract_variables(solutions[objectives.index(min(objectives, key=lambda i: abs(i[0]-i[1])))].variables), self.ensemble_)

        pareto_set, fitnesses = self._genetic_optimalisation(optimalisation_type='quality_single')
        self._ensemble_quality_single = self.get_group(pareto_set[fitnesses.index(max(fitnesses, key=itemgetter(0)))],
                                                       self.ensemble_)
        # pareto_set, fitnesses = self._genetic_optimalisation(optimalisation_type='diversity_single')
        # self._ensemble_diversity_single = self.get_group(pareto_set[fitnesses.index(min(fitnesses, key=itemgetter(0)))],
        #                                                  self.ensemble_)

        pareto_set, fitnesses = self._genetic_optimalisation(optimalisation_type='precision_single')
        self._ensemble_precision_single = self.get_group(pareto_set[fitnesses.index(max(fitnesses, key=itemgetter(0)))],
                                                       self.ensemble_)
        pareto_set, fitnesses = self._genetic_optimalisation(optimalisation_type='recall_single')
        self._ensemble_recall_single = self.get_group(pareto_set[fitnesses.index(max(fitnesses, key=itemgetter(0)))],
                                                         self.ensemble_)




    def set_ensemble(self, ensemble_type='quality'):
        if ensemble_type == 'quality':
            self.ensemble_ = self._ensemble_quality
        elif ensemble_type == 'diversity':
            self.ensemble_ = self._ensemble_diversity
        elif ensemble_type == 'balanced':
            self.ensemble_ = self._ensemble_balanced
        elif ensemble_type == 'diversity_single':
            self.ensemble_ = self._ensemble_diversity_single
        elif ensemble_type == 'quality_single':
            self.ensemble_ = self._ensemble_quality_single
        elif ensemble_type == 'precision_single':
            self.ensemble_ = self._ensemble_precision_single
        elif ensemble_type == 'recall_single':
            self.ensemble_ = self._ensemble_recall_single
        print('stop')

    def get_ensemble_composition(self):
        types = [type(cls) for cls in self.ensemble_]
        return Counter(types)

    @staticmethod
    def subsample(X, y, n_sample=None, ratio=1.0):
        if n_sample is None:
            n_sample = round(len(X) * ratio)
        sample_X = np.empty((n_sample, X.shape[1]))
        sample_y = np.empty((n_sample, 1))
        for i in range(n_sample):
            index = randrange(len(X))
            sample_X[i, :] = X[index, :]
            sample_y[i] = y[index]
        return sample_X, sample_y

    @staticmethod
    def stratified_bagging(X, y, ratio=1.0):
        n_sample = round(len(X) * ratio)
        labels = np.unique(y)
        class_n_sample = [round(n_sample*(len(np.where(y == l)[0])/len(y))) for l in labels]

        class_samples = []
        class_samples_label = []

        for class_sample, label in zip(class_n_sample, labels):
            X_samples, y_samples = MCE.subsample(X[np.where(y == label)[0]], y[np.where(y == label)[0]], n_sample=class_sample)
            class_samples.append(X_samples)
            class_samples_label.append(y_samples)

        X_sample = np.concatenate(class_samples)
        y_sample = np.concatenate(class_samples_label)
        return X_sample, y_sample

    def fit(self, X, y):



        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
        for train_index, test_index in sss.split(X, y):
            self._X_train, self._X_valid = X[train_index], X[test_index]
            self._y_train, self._y_valid = y[train_index], y[test_index]

        # rus = RandomUnderSampler(random_state=42)
        ros = RandomOverSampler(random_state=13)

        for e in self._base_estimator_pool:
            for i in range(self._no_bags):
                #Stratified Bagging
                # X_sample, y_sample = MCE.stratified_bagging(self._X_train, self._y_train, 0.4)
                # new_e = clone(e)
                # new_e.fit(X_sample, y_sample)
                # self.ensemble_.append(new_e)

                #Random Subspace
                # new_e = clone(e)
                # n = randint(1, X.shape[1])
                # sample_index = sample(range(0,X.shape[1]), n)
                # wrap_e = RandomSubspaceClassifierWrapper(new_e, sample_index)
                # wrap_e.fit(X, y)
                # self.ensemble_.append(wrap_e)

                # Random Undersampling
                try:
                    X_sample, y_sample = MCE.subsample(self._X_train, self._y_train, ratio=0.4)
                    X_sample_rus, y_sample_rus = ros.fit_resample(X_sample, y_sample)

                    if len(X_sample_rus) <= 5:
                        raise Exception()
                    new_e_rus = clone(e)
                    new_e_rus.fit(X_sample_rus, y_sample_rus)
                    self.ensemble_.append(new_e_rus)
                except:
                    pass


        self._y_predict = np.array([member_clf.predict(self._X_valid) for member_clf in self.ensemble_])
        # self._pairwise_diversity_stats = np.ones((len(self.ensemble_), len(self.ensemble_)))
        # self._get_pairwise_Q_stat()
        self._prune()
        for m in self.ensemble_:
            m.fit(self.X_, self.y_)

    def score(self, X, y, sample_weight=None):
        prediction = self.predict(X)
        return sum(prediction == y) / len(y)

    def ensemble_support_matrix(self, X):
        """ESM."""
        try:
            result = np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])
        except:
            print("UPS")

        return result

    def predict_proba(self, X):
        """Aposteriori probabilities."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Weight support before acumulation
        weighted_support = self.ensemble_support_matrix(X)

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
