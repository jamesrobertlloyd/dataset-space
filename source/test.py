import os
from pylab import *
import numpy as np

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
	a = np.random.rand(n)
	#print a
	order = np.random.permutation(n)[:k]
	X = []
	y = []
	A = np.random.rand(n,k)
	for i in xrange(p):
		x = (2*np.random.rand(n) - 1)[order]
		temp = np.asarray([np.dot(A[j,:], x) for j in range(n)])
		if len(X) == 0:
			X = temp
		else:
			X = np.vstack((X, temp))
		#print X
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
X,c=  yield_points(2,1,10)
print X
print type(c)
for i in xrange(len(c)):
	if c[i] == 1:
		temp = 'o'
	else:
		temp = 'x'
	scatter(X[i][0], X[i][1], marker=temp)


show()