'''
Simple interfaces to models

James Robert Lloyd 2013
'''

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

import numpy as np

class GaussianNaiveBayes_c():

    def __init__(self):
        self._model = GaussianNB()

    def description(self):
        return 'GNB'
        
    def predict_p(self, X_train, y_train, X_test):
        return self._model.fit(X_train, y_train).predict_proba(X_test)[:,-1]  
        
class Tree_c():

    def __init__(self, min_samples_leaf=5, criterion='gini'):
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self._model = DecisionTreeClassifier(min_samples_leaf=self.min_samples_leaf, criterion=self.criterion)

    def description(self):
        return 'Tree %s %s' % (self.min_samples_leaf, self.criterion)

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
        
class GBM_c():

    def __init__(self, n_estimators=500, learn_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self._model = GradientBoostingClassifier(loss='deviance', n_estimators=n_estimators, learning_rate=learn_rate, max_depth=max_depth)

    def description(self):
        return 'GBM %s %s %s' % (self.n_estimators, self.learn_rate, self.max_depth)

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

class Linear_SVM_c():

    def __init__(self, loss='l2', penalty='l2'):
        self.loss = loss
        self.penalty = penalty
        self._model = LinearSVC(penalty=penalty, loss=loss)

    def description(self):
        return 'Linear SVM %s %s' %(self.penalty, self.loss)

    def predict_p(self, X_train, y_train, X_test):
        return self._model.fit(X_train, y_train).predict(X_test)


class KNN_c():

    def __init__(self, k=5):
        self.k = k
        self._model = KNeighborsClassifier(n_neighbors=k)

    def description(self):
        return 'KNN %s' % (self.k)

    def predict_p(self, X_train, y_train, X_test): 
        self._model.fit(X_train, y_train)
        # Compute empirical probabilities
        return np.array([np.mean(y_train[self._model.kneighbors(X_test[i,:])[1]]==1) for i in range(X_test.shape[0])])
        
list_of_classifiers = [GaussianNaiveBayes_c()] + \
                      [KNN_c(k) for k in [4,8,12,16,20,24,28,32,50,75,100]] + \
                      [LogisticRegression_c(C=C, penalty=penalty) for C in [0.001, 0.01, 0.1, 1, 10, 100] for penalty in ['l1', 'l2']] + \
                      [SGD_c(loss=loss, penalty=penalty) for loss in ['log'] for penalty in ['l1', 'l2', 'elasticnet']] + \
                      [RandomForest_c(n_tree, criterion) for n_tree in [100,200,300,400,500] for criterion in ['gini', 'entropy']] + \
                      [Tree_c(min_samples_leaf, criterion) for min_samples_leaf in [1,5,10,20,50] for criterion in ['gini', 'entropy']] + \
                      [GBM_c(n_estimators, learn_rate, max_depth) for n_estimators in [100,300,500] for learn_rate in [0.0001,0.001,0.01,0.1,1] for max_depth in [1,3,5]] + \
                      [GBM_c(n_estimators, learn_rate, max_depth) for n_estimators in [10,25,50] for learn_rate in [0.0001,0.01,1] for max_depth in [1,2]] + \
                      [Linear_SVM_c(loss=loss, penalty='l2') for loss in ['l1', 'l2']]