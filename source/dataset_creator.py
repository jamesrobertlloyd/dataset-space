import scipy.io as sp
import numpy as np
import os

def create_random_dataset(path, dims, n):
	d = {}
	d['X'] = np.random.rand(n, dims)
	d['y'] = [2**(round(x)) - 1 for x in np.random.rand(n,1)]
	print d
	file_name = 'random' + '_' + str(n) + '_' + str(dims) + '.mat'
	sp.savemat(file_name, d)

create_random_dataset(0, 5,150)
