'''Routines for I/O and creating data files'''

import os
import scipy.io
import numpy as np
import pickle

#### I/O

def load_dictionary(file_name):
    '''Loads either a .mat or .pickle'''
    extension = os.path.splitext(file_name)[-1]
    if extension == '.mat':
        return scipy.io.loadmat(file_name, squeeze_me=True)
    elif extension == '.pickle':
        pickle_file = open(file_name, 'rb')
        data = pickle.load(pickle_file)
        pickle_file.close()
        return data
    else:
        raise Exception('Unrecognised data file extension')
        
def all_data_files(data_dir):
    """Produces list of all .mat, .pickle and .csv files in a directory - returns absolute paths"""
    return [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir) if (file_name[-4:] == '.mat') or (file_name[-7:] == '.pickle') or (file_name[-4:] == '.csv')]
        
#### Processing

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
