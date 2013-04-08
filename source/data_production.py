'''
Routines to coordinate model x dataset evaluation

James Robert Lloyd 2013
'''

#### TODO

# - Do I want to standardise inputs or might I risk destroying structure?
#   - If so, empirical quantiles or zero mean and standard deviation?

import numpy as np
import os.path
import time
import cloud.mp
import psutil

from utils.pyroc import AUC
from utils.data import load_dictionary, split_into_folds, all_data_files, standardise_inputs, standardise_outputs
import models

#### Utilities

def exp_param_defaults(exp_params={}):
    '''Sets all missing parameters to their default values'''
    defaults = {'methods' : models.list_of_classifiers,
                'data_dir' : '../data/class',
                'sleep_time' : 2, # Sleep time between experiments, to prevent cloud communication bottlenecks
                'save_dir' : '../results/class/default',
                'multithread' : True,
                'overwrite' : False,
                'max_job_time' : 1,
                'max_cpu_percent' : 80
                }
    # Iterate through default key-value pairs, setting all unset keys
    for key, value in defaults.iteritems():
        if not key in exp_params:
            exp_params[key] = value
    return exp_params
    
def exp_params_to_str(exp_params):
    result = "Running experiment:\n"
    for key, value in exp_params.iteritems():
        result += "%s = %s,\n" % (key, value)
    return result
    
#### Experiment coordination

def evaluate_and_save(method, data_file, save_file_name):
    data = split_into_folds(standardise_outputs(standardise_inputs(load_dictionary(data_file))))
    score = np.mean([AUC(method.predict_p(fold['X_train'], fold['y_train'], fold['X_test']), fold['y_test']) for fold in data['folds']])
    save_file_dir = os.path.split(save_file_name)[0]
    if not os.path.isdir(save_file_dir):
        os.makedirs(save_file_dir)
    with open(save_file_name, 'w') as save_file:
        save_file.write('%s' % score)
    
def evaluate_all(exp_params):
    job_ids = []
    for data_file in sorted(all_data_files(exp_params['data_dir'])):
        #data = split_into_folds(standardise_inputs(load_dictionary(data_file)))
        data_name = os.path.splitext(os.path.basename(data_file))[0]
        for method in exp_params['methods']:
            save_file_name = os.path.join(exp_params['save_dir'], method.description(), data_name + '.score')
            if exp_params['overwrite'] or (not os.path.isfile(save_file_name)):
                while psutil.cpu_percent() > exp_params['max_cpu_percent']:
                    time.sleep(10)
                print 'Running %s %s' % (data_name, method.description())
                if exp_params['multithread']:
                    job_ids.append(cloud.mp.call(evaluate_and_save, method, data_file, save_file_name, _max_runtime=exp_params['max_job_time']))
                else:
                    evaluate_and_save(method, data_file, save_file_name)
                time.sleep(exp_params['sleep_time'])
            else:
                print 'Skipping %s %s' % (data_name, method.description()) 
    if exp_params['multithread']:
        print 'Waiting for all jobs to complete'
        cloud.mp.join(job_ids, ignore_errors=True)
    print 'Finished'
            
#### Interface
            
def test():
    evaluate_all(exp_param_defaults({'methods' : models.list_of_classifiers}))
            
def try_all_datasets():
    evaluate_all(exp_param_defaults({'methods' : [models.GaussianNaiveBayes_c()], 'sleep_time' : 0, 'multithread' : False}))
