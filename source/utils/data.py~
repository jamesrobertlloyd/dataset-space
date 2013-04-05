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
        
def load_network_cv_data(file_name):
    data = load_dictionary(file_name)
    observed = list(zip(data['train_i'].flat, data['train_j'].flat, data['train_v'].flat))
    missing  = list(zip(data['test_i'].flat,  data['test_j'].flat,  data['test_v'].flat))
    truth = list(data['test_v'].flat)
    return {'observations' : observed, 'missing' : missing, 'truth' : truth}
        
def load_cold_start_data(file_name):
    data = load_dictionary(file_name)
    observed = {'social' : list(zip(data['social_train_i'].flat, data['social_train_j'].flat, data['social_train_v'].flat)),
                'collab' : list(zip(data['collab_train_i'].flat, data['collab_train_j'].flat, data['collab_train_v'].flat))}
    missing  = {'collab' : list(zip(data['collab_test_i'].flat,  data['collab_test_j'].flat,  data['collab_test_v'].flat))}
    truth = list(data['collab_test_v'].flat)
    return {'observations' : observed, 'missing' : missing, 'truth' : truth}
    
#### Data file creation

def create_2_clique_data():
    pass
