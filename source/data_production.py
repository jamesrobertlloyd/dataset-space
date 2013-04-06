'''
Routines to coordinate model x dataset evaluation

James Robert Lloyd 2013
'''

import numpy as np

from utils.pyroc import AUC
from utils.data import load_dictionary
import models

def evaluate_method(method, data):
    return np.mean([AUC(method().predict_p(fold['X_train'], fold['y_train'], fold['X_test']), fold['y_test']) for fold in data['folds']])
    
def split_into_folds(data, n=5, seed=0):
    np.random.seed(seed)
    perm = np.random.permutation(range(data['X'].shape[0]))
    
    folds = []
    for fold in range(n):
        if fold == 0:
            train_i = perm[0:np.floor((n-1)*len(perm)/n)]
            test_i  = perm[np.floor((n-1)*len(perm)/n):]
            folds.append({'X_train' : data['X'][train_i,], 'y_train' : data['y'][train_i], 'X_test' : data['X'][test_i,], 'y_test' : data['y'][test_i]})
        elif fold == n-1:
            train_i = perm[np.floor(len(perm)/n):]
            test_i  = perm[:np.floor(len(perm)/n)]
            folds.append({'X_train' : data['X'][train_i,], 'y_train' : data['y'][train_i], 'X_test' : data['X'][test_i,], 'y_test' : data['y'][test_i]})
        else:
            train_i = list(perm[:np.floor(fold*len(perm)/n)]) + list(perm[np.floor((fold+1)*len(perm)/n):])
            test_i  = perm[np.floor(fold*len(perm)/n):np.floor((fold+1)*len(perm)/n)]
            folds.append({'X_train' : data['X'][train_i,], 'y_train' : data['y'][train_i], 'X_test' : data['X'][test_i,], 'y_test' : data['y'][test_i]})
            
    return {'folds' : folds}
    
def evaluate_all(methods, data_files):
    for data_file in data_files:
        data = split_into_folds(load_dictionary(data_file))
        for method in methods:
            score = evaluate_method(method, data)
            print '%s %s %f' % (data_file, method().description(), score)
            
def test():
    methods = [models.GaussianNaiveBayes_c, models.LogisticRegression_c, models.RandomForest_c]
    data_files = ['../data/class/sonar.mat', '../data/class/ionosphere.mat']
    evaluate_all(methods, data_files)
