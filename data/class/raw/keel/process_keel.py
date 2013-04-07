import numpy as np
import scipy.io
import os.path

for file_name in [a_file for a_file in sorted(os.listdir('./')) if a_file[-4:] == '.dat']:
    print 'Processing %s' % file_name
    data = np.genfromtxt(file_name, delimiter=',')
    X = data[:,:-1]
    y = data[:,-1]
    y[y==max(y)] = 1
    y[y==min(y)] = -1
    save_file_name = file_name[:-4] + '.mat'
    scipy.io.savemat(save_file_name, {'X' : X, 'y' : y})
