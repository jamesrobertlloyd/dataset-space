"""
Some properties of datasets

Rishabh Bhargava 2014
"""

import numpy as np

from scipy.stats import kurtosis

class Kurtosis_p():

	def __init__(self, agg_method='avg'):
		self.agg_method = agg_method

	def description(self):
		return "Kurtosis %s" % self.agg_method

	def get_stat(self, X_train):
		if self.agg_method == 'avg':
			return np.mean([kurtosis(X_train[:,i]) for i in range(len(X_train[0]))])
		else:
			return max([kurtosis(X_train[:,i]) for i in range(len(X_train[0]))])

class Num_rows():

	def description(self):
		return "Num_rows"

	def get_stat(self, X_train):
		return len(X_train)

class Num_columns():

	def description(self):
		return "Num_colms"

	def get_stat(self, X_train):
		return len(X_train[0])

list_of_properties = [Kurtosis_p(agg) for agg in ['avg', 'max']] +[Num_rows(), Num_columns()]