'''
Routines to evaluate results

James Robert Lloyd 2013
'''

import os
import numpy as np

from sklearn.mixture import GMM, VBGMM, DPGMM
from sklearn.decomposition import PCA, FactorAnalysis

from pylab import *
from scipy import *

default_dir = '../results/class/default/'

#### Utilities

def permutation_indices(data):
    return sorted(range(len(data)), key = data.__getitem__)
    
def pretty_scatter(x, y, color, radii, labels):
    for i in range(len(x)):
        text(x[i], y[i], labels[i], size=6, horizontalalignment='center')
    sct = scatter(x, y, c=color, s=radii, linewidths=1, edgecolor='w')
    sct.set_alpha(0.75)

#### Interface

def create_csv_summary(results_dir):
    # Loop over model folders
    method_descriptions = [adir for adir in sorted(os.listdir(results_dir)) if os.path.isdir(os.path.join(results_dir, adir)) and not (adir.startswith('Kurt')) and not (adir.startswith('GBM 100')) and not (adir.startswith('GBM 300'))]
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
            else:
                data_array[i, j] = np.NAN
    print 'Saving array'
    np.savetxt(os.path.join(results_dir, 'summary.csv'), data_array, delimiter=',')
    with open(os.path.join(results_dir, 'methods.csv'), 'w') as save_file:
        save_file.write('\n'.join(method_descriptions))
    with open(os.path.join(results_dir, 'datasets.csv'), 'w') as save_file:
        save_file.write('\n'.join(data_names))
    return data_array
    
def plot_ordered_array(results_dir):
    # Load array
    data_array = np.genfromtxt(os.path.join(results_dir, 'summary.csv'), delimiter=',')
    # Mask the NANs
    mdat = np.ma.masked_array(data_array,np.isnan(data_array))
    # Display with ordered rows and columns
    imshow(data_array[permutation_indices(list(np.mean(mdat, axis=1).data))][:,permutation_indices(list(np.mean(mdat, axis=0).data))])
    show()
    
def save_GPLVM_data(results_dir):
    # Load array
    data_array = np.transpose(np.genfromtxt(os.path.join(results_dir, 'summary.csv'), delimiter=','))
    # Setup GPLVM
    (N, D) = data_array.shape
    Q = 2 # Latent dimensionality
    k = GPy.kern.rbf(Q, ARD=True) + GPy.kern.white(Q, 0.00001) #### TODO - is this good usage?
    # Fit model
    #m = GPy.models.Bayesian_GPLVM(Y=data_array, Q=Q, init='PCA', kernel = k, M=n_pseudo_points)
    m = GPy.models.GPLVM(Y=data_array, Q=Q, init='PCA', kernel = k)
    m.ensure_default_constraints()
    m.optimize_restarts(robust=True)
    # Save fit
    np.savetxt(os.path.join(results_dir, 'GPLVM-datasets-2.csv'), m.X, delimiter=',')

def save_PCA_data(results_dir):
	# Load array
	data_array = np.transpose(np.genfromtxt(os.path.join(results_dir,'summary.csv'),delimiter=','))
	#print data_array
	(N,D) = data_array.shape
	print "number of datasets %d, number of methods %d" %(N,D)
	pca = PCA(n_components = 2)
	x_new = pca.fit_transform(data_array)
	print (pca.explained_variance_ratio_)
	print x_new.shape
	np.savetxt(os.path.join(results_dir, 'PCA-datasets-2.csv'), x_new, delimiter=',')

def save_PCA_method(results_dir):
	# Load array
	data_array = np.genfromtxt(os.path.join(results_dir,'summary.csv'),delimiter=',')
	(N,D) = data_array.shape
	print "number of methods %d, number of datasets %d" %(N,D)
	pca = PCA(n_components = 2)
	x_new = pca.fit_transform(data_array)
	print (pca.explained_variance_ratio_)
	print x_new.shape
	np.savetxt(os.path.join(results_dir,'PCA-methods-2.csv'), x_new, delimiter=',')
    
def plot_GPLVM_data(results_dir, method_index=0):
    # Load relevant datasets
    data_array = np.genfromtxt(os.path.join(results_dir, 'summary.csv'), delimiter=',')
    X = (np.genfromtxt(os.path.join(results_dir, 'GPLVM-datasets-2.csv'), delimiter=','))
    datasets = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'datasets.csv'), 'r').readlines()]
    methods = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'methods.csv'), 'r').readlines()]
    # Plot
    clf()
    pretty_scatter(X[:,0], X[:,1], data_array[method_index,:], 200*np.ones(X[:,0].shape), datasets)
    xlabel('Dimension 1')
    ylabel('Dimension 2')
    title('Performance under %s' % methods[method_index])
    colorbar()
    show()

def plot_PCA_data(results_dir, method_index=[26,46,53,66]):
	# Load relevant datasets
	data_array = np.genfromtxt(os.path.join(results_dir,'summary.csv'),delimiter=',')
	X = (np.genfromtxt(os.path.join(results_dir, 'PCA-datasets-2.csv'), delimiter=','))
	datasets = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'datasets.csv'), 'r').readlines()]
	methods = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'methods.csv'), 'r').readlines()]
	# Plot
	figure()
	for i in xrange(len(method_index)):
		subplot(len(method_index)/2, len(method_index)/2, i)
		pretty_scatter(X[:,0], X[:,1], data_array[method_index[i]-1,:], 200*np.ones(X[:,0].shape), datasets)
		xlabel('Dimension 1')
		ylabel('Dimension 2')
		title('Performance under %s' % methods[method_index[i]-1])
		colorbar()
	show()

def plot_PCA_method_data(results_dir, dataset_index =[5]):
	data_array = np.transpose(np.genfromtxt(os.path.join(results_dir,'summary.csv'),delimiter=','))
	X = (np.genfromtxt(os.path.join(results_dir, 'PCA-methods-2.csv'), delimiter=','))
	datasets = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'datasets.csv'), 'r').readlines()]
	methods = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'methods.csv'), 'r').readlines()]
	figure()
	for i in xrange(len(dataset_index)):
		subplot(1, 1, i)
		pretty_scatter(X[:,0], X[:,1], data_array[dataset_index[i]-1,:], 200*np.ones(X[:,0].shape), methods)
		xlabel('Dimension 1')
		ylabel('Dimension 2')
		title('Performance under %s' % datasets[dataset_index[i]-1])
		colorbar()
	show()
    
def plot_GPLVM_data_cluster(results_dir, n_clusters=None, VB=False):
    # Load relevant datasets
    data_array = np.genfromtxt(os.path.join(results_dir, 'summary.csv'), delimiter=',')
    X = (np.genfromtxt(os.path.join(results_dir, 'GPLVM-datasets-2.csv'), delimiter=','))
    datasets = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'datasets.csv'), 'r').readlines()]
    methods = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'methods.csv'), 'r').readlines()]
    # Fit a mixture model
    if n_clusters is None:
        m = DPGMM()
    elif VB:
        m = VBGMM(alpha = 10, n_components=n_clusters)
    else:
        m = GMM(n_components=n_clusters, n_init=100)
    m.fit(data_array.T)
    clusters = m.predict(data_array.T)
    # Plot
    #clf()
    figure(1)
    pretty_scatter(X[:,0], X[:,1], clusters, 200*np.ones(X[:,0].shape), datasets)
    xlabel('Dimension 1')
    ylabel('Dimension 2')
    if n_clusters is None:
        title('CRP MoG')
    elif VB:
        title('%d clusters with VB' % n_clusters)
    else:
        title('%d clusters with EM' % n_clusters)
    show()
    
def produce_co_clustering_table(results_dir=default_dir, model_clusters=6, data_clusters=6):
    # Load relevant datasets
    data_array = np.genfromtxt(os.path.join(results_dir, 'summary.csv'), delimiter=',')
    X = (np.genfromtxt(os.path.join(results_dir, 'GPLVM-datasets-2.csv'), delimiter=','))
    datasets = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'datasets.csv'), 'r').readlines()]
    methods = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'methods.csv'), 'r').readlines()]
    # Fit mixture models
    m_model = GMM(n_components=model_clusters, n_init=20)
    m_model.fit(data_array)
    model_C = m_model.predict(data_array)
    m_data = GMM(n_components=data_clusters, n_init=20)
    m_data.fit(data_array.T)
    data_C = m_data.predict(data_array.T)
    # Produce table
    performance_table = np.zeros((model_clusters, data_clusters))
    for model_cluster in range(model_clusters):
        # print model clusters
        print 'Model cluster %d' % (model_cluster + 1)
        for (i, method) in enumerate(methods):
            if model_C[i] == model_cluster:
                print method
        for data_cluster in range(data_clusters):
            performance_table[model_cluster, data_cluster] = np.mean(data_array[model_C==model_cluster][:,data_C==data_cluster])
    # Print table
    print performance_table

def factor_analysis(results_dir):
	data_array = np.transpose(np.genfromtxt(os.path.join(results_dir,'summary.csv'),delimiter=','))
	fa = FactorAnalysis(n_components = 2)
	new_array = fa.fit_transform(data_array)
	print fa.get_covariance().shape
	print new_array
	np.savetxt(os.path.join(results_dir,'FA-datasets-2.csv'), new_array, delimiter=',')

def factor_analyses(results_dir):
	data_array = np.genfromtxt(os.path.join(results_dir,'summary.csv'),delimiter=',')
	fa1 = FactorAnalysis(n_components = 1)
	new_array_gbm = fa1.fit_transform(np.transpose(data_array[range(15)]))
	print new_array_gbm.shape
	fa2 = FactorAnalysis(n_components = 1)
	new_array_tree = fa2.fit_transform(np.transpose(data_array[range(41,51) + range(54,64)]))
	print new_array_tree.shape

	fa3 = FactorAnalysis(n_components = 1)
	new_array_lin = fa3.fit_transform(np.transpose(data_array[range(27,41) + range(51,54)]))

	fa4 = FactorAnalysis(n_components = 1)
	new_array_knn = fa4.fit_transform(np.transpose(data_array[range(16,27)]))

	datasets = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'datasets.csv'), 'r').readlines()]
	methods = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'methods.csv'), 'r').readlines()]
	figure()
	pretty_scatter(new_array_tree, [1 for x in range(115)], data_array[46], 200*np.ones(new_array_tree.shape), datasets)


	figure()
	scatter(new_array_gbm, new_array_tree)
	#for i in xrange(len(datasets)):
	#	annotate(datasets[i], xy = (new_array_gbm[i], new_array_tree[i]))
	xlabel('GBM')
	ylabel('Tree')

	figure()
	scatter(new_array_lin, new_array_tree)
	xlabel('Linear')
	ylabel('Tree')

	figure()
	scatter(new_array_knn, new_array_tree)
	xlabel('KNN')
	ylabel('Tree')

	show()

def plot_FA_data(results_dir, method_index=1):
	data_array = np.genfromtxt(os.path.join(results_dir,'summary.csv'),delimiter=',')
	X = (np.genfromtxt(os.path.join(results_dir, 'FA-datasets-2.csv'), delimiter=','))
	datasets = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'datasets.csv'), 'r').readlines()]
	methods = [line.rstrip('\n') for line in open(os.path.join(results_dir, 'methods.csv'), 'r').readlines()]

	figure()
	pretty_scatter(X[:,0], X[:,1], data_array[method_index-1,:], 200*np.ones(X[:,0].shape), datasets)
	xlabel('Dimension 1')
	ylabel('Dimension 2')
	title('Performance under %s' % methods[method_index-1])
	colorbar()
	show()

#create_csv_summary(default_dir)
#plot_ordered_array('../results/class/default/')
#print permutation_indices([3,4,0,1,2])

#factor_analysis(default_dir)
#plot_FA_data(default_dir)

factor_analyses(default_dir)

#plot_GPLVM_data_cluster('../results/class/without_gbm/',method_index)
#save_PCA_data(default_dir)

#save_PCA_method(default_dir)
#plot_PCA_method_data(default_dir)

#method_index=[10,16,23,32]
#plot_PCA_data(default_dir, method_index)
#method_index=[6,16,26,36]

#plot_PCA_data(default_dir,method_index)
#method_index = [71,78,26,89]
#plot_PCA_data(default_dir,method_index)
