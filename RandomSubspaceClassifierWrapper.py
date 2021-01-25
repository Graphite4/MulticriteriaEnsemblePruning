class RandomSubspaceClassifierWrapper():

    def __init__(self, classifier, column_index):
        self._classifier = classifier
        self._col_ind = column_index

    def fit(self, X, y):
        self._classifier.fit(X[:,self._col_ind], y)

    def predict_proba(self, X):
        return self._classifier.predict_proba(X[:, self._col_ind])

    def predict(self, X):
        return self._classifier.predict(X[:,self._col_ind])

    def get_type(self):
        return type(self._classifier)