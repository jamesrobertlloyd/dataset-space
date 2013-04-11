'''
Routines to evaluate results

James Robert Lloyd 2013
'''

import os
import numpy as np
import GPy
from sklearn.mixture import GMM, VBGMM, DPGMM

from pylab import *
from scipy import *

default_dir = '../results/class/without_gbm/'

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
    # Display with orderded rows and columns
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
    clf()
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
