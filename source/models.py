'''
Simple interfaces to models

James Robert Lloyd 2013
'''

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class GaussianNaiveBayes_c():

    def __init__(self):
        pass

    def description(self):
        return 'GNB'
        
    def predict_p(self, X_train, y_train, X_test):
        return GaussianNB().fit(X_train, y_train).predict_proba(X_test)[:,-1]     
        
class RandomForest_c():

    def __init__(self):
        pass

    def description(self):
        return 'RF'

    def predict_p(self, X_train, y_train, X_test): 
        return RandomForestClassifier(n_estimators=500).fit(X_train, y_train).predict_proba(X_test)[:,-1]

class LogisticRegression_c():

    def __init__(self):
        pass

    def description(self):
        return 'LR'

    def predict_p(self, X_train, y_train, X_test): 
        return RandomForestClassifier(n_estimators=500).fit(X_train, y_train).predict_proba(X_test)[:,-1]
    
