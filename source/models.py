'''
Simple interfaces to models

James Robert Lloyd 2013
'''

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

class GaussianNaiveBayes_c():

    def __init__(self):
        self._model = GaussianNB()

    def description(self):
        return 'GNB'
        
    def predict_p(self, X_train, y_train, X_test):
        return self._model.fit(X_train, y_train).predict_proba(X_test)[:,-1]     
        
class RandomForest_c():

    def __init__(self, n_tree=500, criterion='gini'):
        self.n_tree = n_tree
        self.criterion = criterion
        self._model = RandomForestClassifier(n_estimators=self.n_tree, criterion=self.criterion)

    def description(self):
        return 'RF %s %s' % (self.n_tree, self.criterion)

    def predict_p(self, X_train, y_train, X_test): 
        return self._model.fit(X_train, y_train).predict_proba(X_test)[:,-1]

class LogisticRegression_c():

    def __init__(self, C=None, penalty=None):
        self.C = C
        self.penalty = penalty
        self._model = LogisticRegression(C=C, penalty=penalty)

    def description(self):
        return 'LR %s %s' % (self.penalty, self.C)

    def predict_p(self, X_train, y_train, X_test): 
        return self._model.fit(X_train, y_train).predict_proba(X_test)[:,-1]

class SGD_c():

    def __init__(self, loss=None, penalty=None):
        self.loss = loss
        self.penalty = penalty
        self._model = SGDClassifier(loss=loss, penalty=penalty)

    def description(self):
        return 'SGD %s %s' % (self.penalty, self.loss)

    def predict_p(self, X_train, y_train, X_test): 
        return self._model.fit(X_train, y_train).predict_proba(X_test)[:,-1]
        
#### TODO - Implement K nearest neighbours
        
list_of_classifiers = [GaussianNaiveBayes_c()] + \
                      [LogisticRegression_c(C=C, penalty=penalty) for C in [0.1, 1, 10, 100] for penalty in ['l1', 'l2']] + \
                      [SGD_c(loss=loss, penalty=penalty) for loss in ['hinge', 'modified_huber', 'log'] for penalty in ['l1', 'l2', 'elasticnet']] + \
                      [RandomForest_c(n_tree, criterion) for n_tree in [100,200,300,400,500] for criterion in ['gini', 'entropy']]
    
