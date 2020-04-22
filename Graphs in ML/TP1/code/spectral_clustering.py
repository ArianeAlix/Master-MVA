import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skm
import scipy

from utils import plot_clustering_result, plot_the_bend
from build_similarity_graph import build_similarity_graph
from generate_data import blobs, two_moons, point_and_circle
import ruptures as rpt

def build_laplacian(W, laplacian_normalization=""):
    """
    Compute graph Laplacian.

    :param W: adjacency matrix
    :param laplacian_normalization:  string selecting which version of the laplacian matrix to construct
                                     'unn':  unnormalized,
                                     'sym': symmetric normalization
                                     'rw':  random-walk normalization
    :return: L: (n x n) dimensional matrix representing the Laplacian of the graph
    """    
    L=np.zeros(W.shape)
    
    #We compute the corresponding degrees matrix
    D=np.diag(np.sum(W,axis=0))
    
    if laplacian_normalization=="rw":
        L=np.eye(W.shape[0])- np.dot(np.linalg.inv(D),W)
    
    elif laplacian_normalization=="sym":
        L=np.eye(W.shape[0])- np.dot(np.dot(scipy.linalg.fractional_matrix_power(D,-0.5),W),scipy.linalg.fractional_matrix_power(D,0.5))
        
    else: #Choosing the unnormalized laplacian by default
        L=D-W
    
    return L


def spectral_clustering(L, chosen_eig_indices, num_classes=2):
    """
    :param L: Graph Laplacian (standard or normalized)
    :param chosen_eig_indices: indices of eigenvectors to use for clustering
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.sparse.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    eig_val,eig_vec=scipy.linalg.eig(L)
    
    E = np.diag(np.sort(eig_val)) 
    U = eig_vec[:,np.argsort(eig_val)]

    """
    compute the clustering assignment from the eigenvectors        
    Y = (n x 1) cluster assignments [0,1,...,c-1]                   
    """
    k_mean=KMeans(n_clusters=num_classes)
    
    k_mean.fit(U[:,chosen_eig_indices])
    
    Y = k_mean.labels_
    return Y


def two_blobs_clustering():
    """
    Clustering of two blobs. Used in questions 2.1 and 2.2
    """

    # Get data and compute number of classes
    X, Y = blobs(600, n_blobs=2, blob_var=0.15, surplus=0)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """ 
    
    k = 3
    var = 1.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'
                 
    chosen_eig_indices = np.arange(1,num_classes)   # indices of the ordered eigenvalues to pick

    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)
                 
    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    # Plot results
    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))


def choose_eig_function(eigenvalues):
    """
    Function to choose the indices of which eigenvalues to use for clustering.

    :param eigenvalues: sorted eigenvalues (in ascending order)
    :return: indices of the eigenvalues to use
    """
    eigvals=np.sort(eigenvalues)
    
    #We use a function from the imported ruptures package to find the first jump in values
    algo = rpt.Pelt(jump=1).fit(eigvals)
    step_index=algo.predict(pen=0.001)[0]
    
    #We choose the eigenvalues from index 1 to the index before the jump
    eig_ind = np.arange(1,step_index)
    
    return eig_ind


def spectral_clustering_adaptive(L, num_classes=2):
    """
    :param L: Graph Laplacian (standard or normalized)
    :param chosen_eig_indices: indices of eigenvectors to use for clustering
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.sparse.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    eig_val,eig_vec=scipy.linalg.eig(L)
    
    E = np.diag(np.sort(eig_val)) 
    U = eig_vec[:,np.argsort(eig_val)]
   
    chosen_eig_indices=choose_eig_function(eig_val)
    
    """
    compute the clustering assignment from the eigenvectors        
    Y = (n x 1) cluster assignments [0,1,...,c-1]                   
    """
    k_mean=KMeans(n_clusters=num_classes)
    
    k_mean.fit(U[:,chosen_eig_indices])
    
    Y = k_mean.labels_
    return Y


def find_the_bend():
    """
    Used in question 2.3
    :return:
    """

    # the number of samples to generate
    num_samples = 600

    # Generate blobs and compute number of clusters
    X, Y = blobs(num_samples, 4, 0.2)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 2
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'unn'  # either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization


    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)

    """
    compute first 15 eigenvalues and call choose_eig_function() to choose which ones to use. 
    """
    eig_val,eig_vec=scipy.linalg.eig(L)

    
    eigenvalues = np.sort(eig_val)[:15]
    chosen_eig_indices = choose_eig_function(eigenvalues)  # indices of the ordered eigenvalues to pick


    """
    compute spectral clustering solution using a non-adaptive method first, and an adaptive one after (see handout) 
    Y_rec = (n x 1) cluster assignments [0,1,..., c-1]    
    """
    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)
    Y_rec_adaptive = spectral_clustering_adaptive(L, num_classes=num_classes)

    plot_the_bend(X, Y, L, Y_rec, eigenvalues)


def two_moons_clustering():

    # Generate data and compute number of clusters
    X, Y = two_moons(600)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 15
    var = 5.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'

    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L = build_laplacian(W, laplacian_normalization)
    
    eig_val,eig_vec=scipy.linalg.eig(L)

    
    eigenvalues = np.sort(eig_val)[:15]
    chosen_eig_indices = choose_eig_function(eigenvalues)
    
    
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))


def point_and_circle_clustering():

    # Generate data and compute number of clusters
    X, Y = point_and_circle(600)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k =15
    var = 5.0  # exponential_euclidean's sigma^2


    # build laplacian
    W = build_similarity_graph(X, var=var, k=k)
    L_unn = build_laplacian(W, 'unn')
    L_rw = build_laplacian(W, 'rw')
    
    
    #Computing anf choosing the eigenvalues for the two types of Laplacian Matrices
    eig_val_unn,eig_vec_unn=scipy.linalg.eig(L_unn)
    eigenvalues_unn = np.sort(eig_val_unn)[:15]
    chosen_eig_indices_unn = choose_eig_function(eigenvalues_unn)
    
    eig_val_rw,eig_vec_rw=scipy.linalg.eig(L_rw)
    eigenvalues_rw = np.sort(eig_val_rw)[:15]
    chosen_eig_indices_rw = choose_eig_function(eigenvalues_rw)
    

    Y_unn = spectral_clustering(L_unn, chosen_eig_indices_unn, num_classes=num_classes)
    Y_rw = spectral_clustering(L_rw, chosen_eig_indices_rw, num_classes=num_classes)

    plot_clustering_result(X, Y, L_unn, Y_unn, Y_rw, 1)


def parameter_sensitivity():
    """
    A function to test spectral clustering sensitivity to parameter choice.

    Used in question 2.9
    """
    # the number of samples to generate
    num_samples = 500

    """
    Choose parameters
    """
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'unn'

    """
    Choose candidate parameters
    """
    parameter_candidate = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50]  # the number of neighbours for the graph or the epsilon threshold
    parameter_performance = []

    for k in parameter_candidate:
        # Generate data
        X, Y = two_moons(num_samples, 1, 0.02)
        num_classes = len(np.unique(Y))

        W = build_similarity_graph(X, k=k)
        L = build_laplacian(W, laplacian_normalization)
        
        eig_val,eig_vec=scipy.linalg.eig(L)
        eigenvalues = np.sort(eig_val)[:15]
        chosen_eig_indices = choose_eig_function(eigenvalues)

        Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes)

        parameter_performance += [skm.adjusted_rand_score(Y, Y_rec)]

    plt.figure()
    plt.plot(parameter_candidate, parameter_performance)
    plt.title('parameter sensitivity')
    plt.show()

