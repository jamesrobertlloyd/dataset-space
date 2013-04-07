'''
Routines to evaluate results

James Robert Lloyd 2013
'''

import os
import numpy as np

def create_csv_summary(results_dir):
    # Loop over model folders
    method_descriptions = [adir for adir in sorted(os.listdir(results_dir)) if os.path.isdir(os.path.join(results_dir, adir))]
    data_names = []
    data_dictionary = {method_description : {} for method_description in method_descriptions}
    for method_description in method_descriptions:
        print 'Reading %s' % method_description
        data_names = sorted(list(set(data_names + [os.path.splitext(file_name)[0] for file_name in [full_path for full_path in sorted(os.listdir(os.path.join(results_dir, method_description))) if full_path[-6:] == '.score']])))
        for data_name in [file_name for file_name in sorted(os.listdir(os.path.join(results_dir, method_description))) if file_name[-6:] == '.score']:
            with open(os.path.join(results_dir, method_description, data_name), 'rb') as score_file:
                score = float(score_file.read())
            data_dictionary[method_description][os.path.splitext(data_name)[0]] = score
    # Create array
    print 'Creating array'
    data_array = -0.01 * np.ones((len(method_descriptions), len(data_names)))
    for (i, method_description) in enumerate(method_descriptions):
        for (j, data_name) in enumerate(data_names):
            if (method_description in data_dictionary) and (data_name in data_dictionary[method_description]):
                data_array[i, j] = data_dictionary[method_description][data_name]
    print 'Saving array'
    np.savetxt(os.path.join(results_dir, 'summary.csv'), data_array, delimiter=',')
    #### TODO - save names and methods
    return data_array
