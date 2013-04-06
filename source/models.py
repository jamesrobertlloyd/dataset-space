'''
Simple interfaces to models

James Robert Lloyd 2013
'''

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class GaussianNaiveBayes_c():

    def __init__(self):
        self._model = GaussianNB()

    def description(self):
        return 'GNB'
        
    def predict_p(self, X_train, y_train, X_test):
        return self._model.fit(X_train, y_train).predict_proba(X_test)[:,-1]     
        
class RandomForest_c():

    def __init__(self, n_estimators=500):
        self.n_estimators = n_estimators
        self._model = RandomForestClassifier(n_estimators=self.n_estimators)

    def description(self):
        return 'RF %s' % self.n_estimators

    def predict_p(self, X_train, y_train, X_test): 
        return self._model.fit(X_train, y_train).predict_proba(X_test)[:,-1]

class LogisticRegression_c():

    def __init__(self):
        self._model = LogisticRegression()

    def description(self):
        return 'LR'

    def predict_p(self, X_train, y_train, X_test): 
        return self._model.fit(X_train, y_train).predict_proba(X_test)[:,-1]
    
