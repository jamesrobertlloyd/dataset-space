import os
from pylab import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn. tree import DecisionTreeClassifier
from utils.data import split_into_folds
from utils.pyroc import AUC
from models import *

'''
This method creates a k dimensional object in an n dimensional space and returns points on it
'''
def create_plane(n, k):
	b = np.random.rand(k,1)
	pars = np.random.rand(100,k)
	a = np.random.rand(n,k)
	points = []
	for i in xrange(100):
		temp = np.zeros(n)
		for j in xrange(k):
			temp = np.add(pars[i][j]*a[:,j], temp)
		points.append(temp)
	return points

'''
This method creates a k dimensional object in an n dimensional space and returns p points around it with correct labels
'''
def yield_points(n,k,p):
	a = np.ones(n)
	#print a
	order = np.random.permutation(n)[:k]
	X = []
	y = []
	A = np.random.rand(n,k)
	#A = np.ones((n,k))
	for i in xrange(p):
		# x = [round()(2*np.random.rand(n) - 1)[order]]
		x = [1 if np.random.rand() > 0.5 else -1 for _ in range(n)]
		temp = np.asarray([np.dot(A[j,:], x) for j in range(n)])
		if len(X) == 0:
			X = temp
		else:
			X = np.vstack((X, temp))
		#print X
		#temp = x
		#type(temp)
		if np.random.rand() < 0.1:
			temp = -temp
		if np.dot(temp,a) > 0:
			y = np.append(y, 1)
		else:
			y = np.append(y, -1)
	return (X,y)


'''
test = create_plane(2,1)
x = [a[0] for a in test]
y = [a[1] for a in test]
'''
##X,y=  yield_points(2,1,10)
#X,y = yield_points(3000,20,2000)
#print X
#print reduce(lambda x,y: x+ y, filter(lambda x: x > 0, y))
'''
for i in xrange(len(y)):
	if y[i] == 1:
		temp = 'o'
	else:
		temp = 'x'
	scatter(X[i][0], X[i][1], marker=temp)


show()
'''

def get_the_plots(tups):
	ratios = []
	rf_scores = []
	lr_scores = []
	tree_scores = []
	for i in xrange(len(tups)):
		n,k,p = tups[i]
		X,y = yield_points(n,k,p)
		d = {}
		d['X'] = X
		d['y'] = y
		data = split_into_folds(d)
		rfmodel = RandomForest_c(10, 'gini')
		score = np.mean([AUC(rfmodel.predict_p(fold['X_train'], fold['y_train'], fold['X_test']), fold['y_test']) for fold in data['folds']])
		rf_scores.append(score)
		print score

		logregmodel = LogisticRegression_c(C=0.1, penalty='l2')
		score = np.mean([AUC(logregmodel.predict_p(fold['X_train'], fold['y_train'], fold['X_test']), fold['y_test']) for fold in data['folds']])
		lr_scores.append(score)
		print score

		dtmodel = Tree_c(10, 'gini')
		score = np.mean([AUC(dtmodel.predict_p(fold['X_train'], fold['y_train'], fold['X_test']), fold['y_test']) for fold in data['folds']])
		tree_scores.append(score)
		print score

		ratios.append(p)
	line1, = plot(ratios,rf_scores,'r', label='RF')
	line1, = plot(ratios, lr_scores, 'g', label='LR')
	line1, = plot(ratios, tree_scores, 'k', label='Tree')
	legend()
	xlabel('Number of points in the dataset')
	ylabel('Performance score')
	show()


get_the_plots([(1000, 1000, 10),(1000, 1000, 20)] + [(1000,1000,21+2*x) for x in range(4)] + [(1000,1000,20+10*(x+1)) for x in range(8)])

#get_the_plots([(100,100,5+ 2*(x+1)) for x in range(30)])

'''d = {}
d['X'] = X
d['y'] = y
data = split_into_folds(d)
rfmodel = RandomForest_c(10, 'gini')
score = np.mean([AUC(rfmodel.predict_p(fold['X_train'], fold['y_train'], fold['X_test']), fold['y_test']) for fold in data['folds']])
print score

logregmodel = LogisticRegression_c(C=0.1, penalty='l2')
score = np.mean([AUC(logregmodel.predict_p(fold['X_train'], fold['y_train'], fold['X_test']), fold['y_test']) for fold in data['folds']])
print score
'''