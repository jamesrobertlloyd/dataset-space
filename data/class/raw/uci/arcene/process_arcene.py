import numpy as np
import scipy.io

X = np.genfromtxt('arcene_train.data', delimiter=' ')
y = np.genfromtxt('arcene_train.labels', delimiter=' ')
scipy.io.savemat('arcene.mat', {'X' : X, 'y' : y})
