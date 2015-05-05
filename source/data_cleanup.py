import os
import shutil
import numpy as np
import scipy
import csv


path = '../data/class/raw/automatic_statistician/' + os.listdir('../data/class/raw/automatic_statistician/')[0]
print path

def load_dictionary(path):
	print path
	try:
		data_array = np.loadtxt(path, delimiter = ',', dtype = float)
	except ValueError:
		try:
			data_array = np.loadtxt(path, delimiter = ',', dtype = float, skiprows = 1)
		except ValueError:
			print path
	print data_array.shape
	print data_array

load_dictionary(path)
