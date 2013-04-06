'''
Routines to coordinate model x dataset evaluation

James Robert Lloyd 2013
'''

import numpy as np
import os.path

from utils.pyroc import AUC
from utils.data import load_dictionary, split_into_folds, all_data_files
import models

def evaluate_method(method, data):
    return np.mean([AUC(method.predict_p(fold['X_train'], fold['y_train'], fold['X_test']), fold['y_test']) for fold in data['folds']])
    
def evaluate_all(methods, data_files):
    for data_file in data_files:
        data = split_into_folds(load_dictionary(data_file))
        for method in methods:
            score = evaluate_method(method, data)
            print '%s %s %f' % (os.path.splitext(os.path.basename(data_file))[0], method.description(), score)
            
def test():
    methods = [models.GaussianNaiveBayes_c(), models.LogisticRegression_c(), models.RandomForest_c(100), models.RandomForest_c(200), models.RandomForest_c(300)]
    data_files = all_data_files('../data/class')
    evaluate_all(methods, data_files)
