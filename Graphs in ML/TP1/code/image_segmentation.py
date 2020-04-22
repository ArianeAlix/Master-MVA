from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
from build_similarity_graph import build_similarity_graph
from spectral_clustering import build_laplacian, spectral_clustering, choose_eig_function

import scipy
import ruptures as rpt


def image_segmentation(input_img='four_elements.bmp'):
    """
    Function to perform image segmentation.

    :param input_img: name of the image file in /data (e.g. 'four_elements.bmp')
    """
    filename = os.path.join('data', input_img)

    X = io.imread(filename)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))

    im_side = np.size(X, 1)
    Xr = X.reshape(im_side ** 2, 3)
    """
    Y_rec should contain an index from 0 to c-1 where c is the     
     number of segments you want to split the image into          
    """

    """
    Choose parameters
    """
    var = 5.0
    k = 80
    
    # We use an unnormalized laplacian, because usually in pictures, we have an homogenous
    # distribution of colors. As a consequence, a clustering aiming to minimize the unbalance between
    # the number of pixels in the different color clusters makes sense
    laplacian_normalization = 'unn'
    
    # We choose to build the similarity graph using a KNN because we do not want to simply
    # cut an edge between colors when their distance is above a threshold, because it might not make
    # sense in a picture where you could observe different contrasts in the different parts
    print("Building the similarity graph using a K-nn, with k="+str(k)+"...")
    W = build_similarity_graph(Xr, var=var, k=k)
    print("Done.")
    
    print("Computing the Laplacian...")
    L = build_laplacian(W, laplacian_normalization)
    print("Done.")
    
    # We choose the eigenvectors based on the eigenvalues of L
    eig_val,eig_vec=scipy.linalg.eig(L)
    eigenvalues = np.sort(eig_val)
    
    
    '''
    We will do clustering for different numbers of clusters
    To do that we will choose the eigenvectors corresponding to
    the 1st jump, 2nd jump etc. and the nb of clusters will correpsond
    to the nb of eigenvectors +1
    '''
    eigvals=eigenvalues
    previous_step=0
    for i in range(3):

        # We use the same function as in choose_eig_function the spectral clustering part, that takes
        # the eigenvectors corrresponding to the positive eigenvalues before a jump in value
        algo = rpt.Pelt(jump=1).fit(eigvals)
        step_index=algo.predict(pen=0.001)[0]
        
        
        #We choose the eigenvalues from index 1 to the index before the jump
        chosen_eig_indices = np.arange(1,step_index+previous_step)    
        
        # the number of indices chosen is linked to the number of clusters we will generate
        num_classes = len(chosen_eig_indices)+1 

        print("Number of clusters :",num_classes)

        Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(X)

        plt.subplot(1, 2, 2)
        Y_rec = Y_rec.reshape(im_side, im_side)
        plt.imshow(Y_rec)

        plt.show()
        
        
        # for the next step using the next jump in eigenvalues
        eigvals=eigvals[step_index:]
        previous_step+=step_index
        

if __name__ == '__main__':
    image_segmentation()
