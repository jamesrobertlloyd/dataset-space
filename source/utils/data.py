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
