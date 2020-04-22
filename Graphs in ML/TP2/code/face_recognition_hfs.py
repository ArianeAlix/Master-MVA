import matplotlib.pyplot as plt
from imageio import imread
import numpy as np
import cv2
import os

from helper import *

from harmonic_function_solution import *


def offline_face_recognition(filters=['GaussianBlur'],plot=True):
    """
    Function to test offline face recognition.
    """

    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))

    frame_size = 96
    # Loading images
    images = np.zeros((100, frame_size ** 2))
    labels = np.zeros(100)

    for i in np.arange(10):
        for j in np.arange(10):
            im = imread("data/10faces/%d/%02d.jpg" % (i, j + 1))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            gray_face = cv2.equalizeHist(gray_face)
            
            # Modofoed to allow us to choose
            for f in filters:
                if f=='GaussianBlur':            
                    gray_face = cv2.GaussianBlur(gray_face, (5, 5), 0)
                if f=='boxFilter':            
                    gray_face = cv2.boxFilter(gray_face,-1, (5, 5))
                if f=='bilateralFilter':            
                    gray_face = cv2.bilateralFilter(gray_face, 2, 10,10, 0) #2=diameterof neighorhood, 10=sigma for color and space
            
            gf = cv2.resize(gray_face, (frame_size, frame_size)).astype(np.float)
            gf -= gf.mean()
            gf /= gf.max()
            gray_face = gf
            # resize the face and reshape it to a row vector, record labels
            images[j * 10 + i] = gray_face.reshape((-1))
            labels[j * 10 + i] = i + 1

    """
     If you want to plot the dataset, set the following variable to True
    """
    plot_the_dataset = False

    if plot_the_dataset:
        plt.figure(1)
        for i in range(100):
            plt.subplot(10, 10, i+1)
            plt.axis('off')
            plt.imshow(images[i].reshape(frame_size, frame_size))
            r='{:d}'.format(i+1)
            if i < 10:
                plt.title('Person '+r)
        plt.show()

    """
    select 4 random labels per person and reveal them  
    Y_masked: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    mlabels = labels.copy()
    for i in range(10):
        mask = np.arange(10)
        np.random.shuffle(mask)
        mask = mask[:6]
        for m in mask:
            mlabels[m * 10 + i] = 0
    
        
    """
     Choose the experiment parameter and compute hfs solution using either soft_hfs or hard_hfs  
    """
    gamma = .95
    var = 50
    eps = 5 #since k!=0 not used
    k = 10
    laplacian_regularization = gamma
    laplacian_normalization = 'rw'
    c_l = 10
    c_u = 5


    # hard or soft HFS
    rlabels = soft_hfs(images, mlabels, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)
    
    #Plot if asked (default =True)
    if plot:
    # Plots #
        plt.subplot(121)
        plt.imshow(labels.reshape((10, 10)))

        plt.subplot(122)
        plt.imshow(rlabels.reshape((10, 10)))
        plt.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

        plt.show()
    return np.equal(rlabels, labels).mean()#accuracy
    
    
    
def offline_face_recognition_augmented(labeled='unlabeled'):

    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    frame_size = 96
    gamma = .95
    nbimgs = 50
    
    # Loading images
    # adding +100 to the nb of row to store the initial data too
    images = np.zeros((10 * nbimgs +100, frame_size ** 2))
    labels = np.zeros(10 * nbimgs +100)
    
    ###
    # Initial data
    for i in np.arange(10):
        for j in np.arange(10):
            im = imread("data/10faces/%d/%02d.jpg" % (i, j + 1))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            gray_face = cv2.equalizeHist(gray_face)
            ray_face = cv2.boxFilter(gray_face,-1, (5, 5))
               
            gf = cv2.resize(gray_face, (frame_size, frame_size)).astype(np.float)
            gf -= gf.mean()
            gf /= gf.max()
            gray_face = gf
            # resize the face and reshape it to a row vector, record labels
            images[j * 10 + i] = gray_face.reshape((-1))
            labels[j * 10 + i] = i + 1
    
    ###
    # Augmented data
    for i in np.arange(10):
        imgdir = "data/extended_dataset/%d" % i
        imgfns = os.listdir(imgdir)
        for j, imgfn in enumerate(np.random.choice(imgfns, size=nbimgs)):
            im = imread("{}/{}".format(imgdir, imgfn))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0, "box": (0, 0, *im.shape[:2])}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            gray_face = cv2.equalizeHist(gray_face)
            ## Box filter instead of Gaussian blur
            gray_face = cv2.boxFilter(gray_face,-1, (5, 5))
            gf = cv2.resize(gray_face, (frame_size, frame_size)).astype(np.float)
            gf -= gf.mean()
            gf /= gf.max()
            gray_face = gf
            
            # resize the face and reshape it to a row vector, record labels
            #adding +100 to the index to store it after the inital data
            images[j * 10 + i +100] = gray_face.reshape((-1))
            labels[j * 10 + i +100] = i + 1

            
    """
     If you want to plot the dataset, set the following variable to True
    """
    plot_the_dataset = False
    if plot_the_dataset:

        plt.figure(1)
        for i in range(10 * (nbimgs + 10)):
            plt.subplot(nbimgs+10,10,i+1)
            plt.axis('off')
            plt.imshow(images[i].reshape(frame_size, frame_size),cmap='gray')
            r='{:d}'.format(i+1)
            if i < 10:
                plt.title('Person '+r)
        plt.show()

        
    """
    select 4 random labels per person and reveal them  
    Y_masked: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    
    if labeled=='labeled':
        #we only mask six out of the 10 first pictures per person (which is only the original dataset)
        mlabels = labels.copy()
        for i in range(10):
            mask = np.arange(10)
            np.random.shuffle(mask)
            mask = mask[:6]
            for m in mask:
                mlabels[m * 10 + i] = 0
                
    elif labeled=='mixed':
        mlabels = labels.copy()
        #same initialization for the original dataset
        for i in range(10):
            mask = np.arange(10)
            np.random.shuffle(mask)
            mask = mask[:6]
            for m in mask:
                mlabels[m * 10 + i] = 0
        #same idea as for the original dataset to keep 40% of the labels per person in the added dataset
        for i in range(10):
            mask = np.arange(nbimgs)
            np.random.shuffle(mask)
            mask = mask[:int(nbimgs*0.6)]
            for m in mask:
                mlabels[m * 10 + i + 100] = 0
                
    else:
        mlabels = labels.copy()
        #same initialization for the original dataset
        for i in range(10):
            mask = np.arange(10)
            np.random.shuffle(mask)
            mask = mask[:6]
            for m in mask:
                mlabels[m * 10 + i] = 0
        #setting all the labels of the added dataset to 0
        for i in range(100,100+10 * nbimgs):
            mlabels[i] = 0
            
    
    
    
    ##Sama parameters and model as in the normal function
    """
     Choose the experiment parameter and compute hfs solution using either soft_hfs or hard_hfs  
    """
    gamma = .95
    var = 50
    eps = 5 #since k!=0 not used
    k = 25
    laplacian_regularization = gamma
    laplacian_normalization = 'rw'
    c_l = 10
    c_u = 5

    # hard or soft HFS
    rlabels = soft_hfs(images, mlabels, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    
    
    """
    Plots
    """
    plt.subplot(121)
    plt.imshow(labels.reshape((-1, 10)))

    plt.subplot(122)
    plt.imshow(rlabels.reshape((-1, 10)))
    plt.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    plt.show()


if __name__ == '__main__':
    offline_face_recognition()
