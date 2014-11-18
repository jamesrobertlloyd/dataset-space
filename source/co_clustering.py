import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score
import os.path

my_file = open('../results/class/default/summary.csv', 'r')
data = []
for line in my_file:
	data.append([float(x) for x in line.split(',')])

data = np.array(data)
print data.shape
plt.imshow(np.array(data), cmap=plt.cm.Blues)
plt.title("Original dataset")

model = SpectralCoclustering(n_clusters=3, random_state=0)
model.fit(data)

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.show()