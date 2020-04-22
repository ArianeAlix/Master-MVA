import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd
from scipy.io import loadmat
import os
from sklearn.preprocessing import OneHotEncoder

from helper import build_similarity_graph, build_laplacian, plot_classification, label_noise, \
                    plot_classification_comparison, plot_clusters, plot_graph_matrix

np.random.seed(50)


def build_laplacian_regularized(X, laplacian_regularization, var=1.0, eps=0.0, k=0, laplacian_normalization=""):
    """
    Function to construct a regularized Laplacian from data.

    :param X: (n x m) matrix of m-dimensional samples
    :param laplacian_regularization: regularization to add to the Laplacian (parameter gamma)
    :param var: the sigma value for the exponential function, already squared
    :param eps: threshold eps for epsilon graphs
    :param k: number of neighbours k for k-nn. If zero, use epsilon-graph
    :param laplacian_normalization: string selecting which version of the laplacian matrix to construct
                                    'unn':  unnormalized,
                                    'sym': symmetric normalization
                                    'rw':  random-walk normalization
    :return: Q (n x n ) matrix, the regularized Laplacian
    """
    # build the similarity graph W
    W = build_similarity_graph(X, var, eps, k)

    """
    Build the Laplacian L and the regularized Laplacian Q.
    Both are (n x n) matrices.
    """
    L = build_laplacian(W, laplacian_normalization)

    # compute Q
    Q = L + laplacian_regularization*np.eye(W.shape[0])

    return Q


def mask_labels(Y, l):
    """
    Function to select a subset of labels and mask the rest.

    :param Y:  (n x 1) label vector, where entries Y_i take a value in [1, ..., C] , where C is the number of classes
    :param l:  number of unmasked (revealed) labels to include in the output
    :return:  Y_masked:
               (n x 1) masked label vector, where entries Y_i take a value in [1, ..., C]
               if the node is labeled, or 0 if the node is unlabeled (masked)
    """
    num_samples = np.size(Y, 0)

    """
     randomly sample l nodes to remain labeled, mask the others   
    """
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    indices = indices[:l]

    Y_masked = np.zeros(num_samples)
    Y_masked[indices] = Y[indices]

    return Y_masked


def hard_hfs(X, Y, laplacian_regularization, var=1, eps=0, k=0, laplacian_normalization=""):
    """
    Function to perform hard (constrained) HFS.

    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n x 1) vector with nodes labels [0, 1, ... , num_classes] (0 is unlabeled)
    :param laplacian_regularization: regularization to add to the Laplacian
    :param var: the sigma value for the exponential function, already squared
    :param eps: threshold eps for epsilon graphs
    :param k: number of neighbours k for k-nn. If zero, use epsilon-graph
    :param laplacian_normalization: string selecting which version of the laplacian matrix to construct
                                    'unn':  unnormalized,
                                    'sym': symmetric normalization
                                    'rw':  random-walk normalization
    :return: labels, class assignments for each of the n nodes
    """

    Cl = np.unique(self.Y)
    num_classes = len(Cl)-1
    
    # Casting to integers to avoid type errors
    Y=Y.astype(int)
    

    """
    Build the vectors:
    l_idx = (l x num_classes) vector with indices of labeled nodes
    u_idx = (u x num_classes) vector with indices of unlabeled nodes
    """

    l_idx=np.where(Y>0)[0]
    u_idx=np.where(Y==0)[0]

    """
    Compute the hfs solution, remember that you can use the functions build_laplacian_regularized and 
    build_similarity_graph    
    
    f_l = (l x num_classes) hfs solution for labeled data. It is the one-hot encoding of Y for labeled nodes.   
    
    example:         
        if Cl=[0,3,5] and Y=[0,0,0,3,0,0,0,5,5], then f_l is a 3x2  binary matrix where the first column codes 
        the class '3'  and the second the class '5'.    
    
    In case of 2 classes, you can also use +-1 labels      
        
    f_u = array (u x num_classes) hfs solution for unlabeled data
    
    f = array of shape(num_samples, num_classes)
    """
    n_l=len(l_idx) # number of labeled points
    n_u=len(u_idx) # number of unlabeled points
    
    f_l = np.zeros((n_l,num_classes))
    for i in range(n_l):
        # for each line representing a labeled X we put 1 in the column corresponding to the label
        # Y[l_idx] : labels of labeled points in order; Y[l_idx][i] : label of ith labeled point
        f_l[i,Y[l_idx][i]-1]=1 #-1 since the labels start at 1
    
    L=build_laplacian_regularized(X,laplacian_regularization, var, eps, k, laplacian_normalization="")
    L_uu=L[u_idx,:][:,u_idx]
    L_ul=L[u_idx,:][:,l_idx]
    
    #f_l is used as y_l in the formula since they should be equal
    f_u = -np.dot(np.dot(np.linalg.inv(L_uu), L_ul), f_l)
    
    
    f = np.zeros((n_l+n_u,num_classes))
    
    f[l_idx,:]=f_l
    f[u_idx,:]=f_u
    

    """
    compute the labels assignment from the hfs solution   
    labels: (n x 1) class assignments [1,2,...,num_classes]    
    """
    labels = np.argmax(f,axis=1)+1 #to compensate for the minus 1 in line 126

    return labels


def two_moons_hfs(dataset):
    """
    HFS for two_moons data.
    """

    """
    Load the data. At home, try to use the larger dataset (question 1.2).    
    """
    # load the data
    in_data = loadmat(os.path.join('data', dataset))
    X = in_data['X']
    Y = in_data['Y'].squeeze()

    # automatically infer number of labels from samples
    num_samples = np.size(Y, 0)
    num_classes = len(np.unique(Y))

    """
    Choose the experiment parameters
    """
    var = 0.1
    eps = 0.1
    k = 10
    laplacian_regularization = 0
    laplacian_normalization = None
    c_l = 100
    c_u = 10

    # number of labeled (unmasked) nodes provided to the hfs algorithm
    l = 4

    # mask labels
    Y_masked = mask_labels(Y, l)

    """
    compute hfs solution using either soft_hfs or hard_hfs
    """
    #labels = hard_hfs(X, Y_masked, laplacian_regularization, var, eps, k, laplacian_normalization)
    labels = soft_hfs(X, Y_masked, c_l , c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    """
    Visualize results
    """
    plot_classification(X, Y, labels,  var=var, eps=0, k=k)
    accuracy = np.mean(labels == np.squeeze(Y))
    
    return accuracy

    
def soft_hfs(X, Y, c_l, c_u, laplacian_regularization, var=1, eps=0, k=0, laplacian_normalization=""):
    """

    Function to perform soft (unconstrained) HFS


    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
    :param c_l: coefficients for C matrix
    :param c_u: coefficients for C matrix
    :param laplacian_regularization:
    :param var:
    :param eps:
    :param k:
    :param laplacian_normalization:
    :return: labels, class assignments for each of the n nodes
    """

    num_samples = np.size(Y, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    """
    Compute the target y for the linear system  
    y = (n x num_classes) target vector 
    l_idx = (l x num_classes) vector with indices of labeled nodes    
    u_idx = (u x num_classes) vector with indices of unlabeled nodes 
    """

    # Casting to integers to avoid type errors
    Y=Y.astype(int)
    
    l_idx=np.where(Y!=0)[0]
    u_idx=np.where(Y==0)[0]
    
    n_l=len(l_idx) # number of labeled points
    n_u=len(u_idx) # number of unlabeled points
    
    # One-hot-encoding for the labeled points
    y= np.zeros((n_l+n_u,num_classes))
    # Adding values only for the lines of labeled data
    for l in l_idx:
        y[l,Y[l]-1]=1 #-1 since the labels start at 1
        

    """
    compute the hfs solution, remember that you can use build_laplacian_regularized and build_similarity_graph
    f = (n x num_classes) hfs solution 
    C = (n x n) diagonal matrix with c_l for labeled samples and c_u otherwise    
    """
    
    diag_c= c_u * (Y==0) + c_l * (Y!=0)
    C=np.diag(diag_c)
    
    L=build_laplacian_regularized(X,laplacian_regularization, var, eps, k, laplacian_normalization="")
    f = np.dot(np.linalg.inv(np.dot(np.linalg.inv(C),L) + np.eye(n_l+n_u)),y)

    """
    compute the labels assignment from the hfs solution 
    labels: (n x 1) class assignments [1, ... ,num_classes]  
    """
    labels = np.argmax(f,axis=1)+1 #to compensate for the minus 1 in line 243

    return labels


def hard_vs_soft_hfs():
    """

    Function to compare hard and soft HFS.
    """
    # load the data
    in_data = loadmat(os.path.join('data', 'data_2moons_hfs.mat'))
    X = in_data['X']
    Y = in_data['Y'].squeeze()

    # automatically infer number of labels from samples
    num_samples = np.size(Y, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1
    
    # randomly sample 20 labels
    l = 20
    # mask labels
    Y_masked = mask_labels(Y, l)

    # Create some noisy labels
    Y_masked[Y_masked != 0] = label_noise(Y_masked[Y_masked != 0], 4)

    """
    choose parameters
    """
    var = 0.1
    eps = 0.1
    k = 10
    laplacian_regularization = 0
    laplacian_normalization = 'rw'
    c_l = 1
    c_u = 10
    
    #print('c_l='+str(c_l),'\nc_u='+str(c_u))

    """
    Compute hfs solution using soft_hfs() and hard_hfs().
    Remember to use Y_masked (the vector with some labels hidden as input and NOT Y (the vector with all labels 
    revealed)
    """
    hard_labels = hard_hfs(X, Y_masked, laplacian_regularization, var, eps, k, laplacian_normalization)
    soft_labels = soft_hfs(X, Y_masked, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    plot_classification_comparison(X, Y, hard_labels, soft_labels, var=var, eps=eps, k=k)
    accuracy = [np.mean(hard_labels == np.squeeze(Y)), np.mean(soft_labels == np.squeeze(Y))]
    return accuracy


if __name__ == '__main__':
    pass


