import os, sys
from PIL import Image
import numpy as np
from os.path import join, getsize
import errno
from os import strerror
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Read the image from the dataset
def read_images(path, sz=None):
    files=os.listdir(path)
    index=0
    imgSet,y=[],[]
    for file in files:
        if not os.path.isdir(file):
            im=Image.open(path+'/'+file)
            # im=im.convert("L") # Here we convert the image to L mode
            if(sz is not None): # if sz is selected, then we need to resize the image.
                im=im.resize(sz)
            imgSet.append(np.asarray(im, dtype=np.uint8))
            y.append(index)
        index=index+1
    #print len(X) just for test
    return [imgSet, y]

# transfer the readed image to matrix to perform PCA
def asMatrix(imgSet):
    if len(imgSet)==0:
        return np.empty((0, imgSet[0].size), dtype=imgSet[0].dtype)
    matrix=np.empty((0, imgSet[0].size), dtype=imgSet[0].dtype)
    for pic in imgSet:
        matrix=np.vstack((matrix, np.asarray(pic).reshape(1,-1)))
    #print mat.shape just for test
    return matrix


def pca(matrix,n_components=0): #default number of eigenfaces=0
    #print X.shape just for test
    [n,d]=matrix.shape
    if(n_components==0):
        n_components=n
    if(n_components>=n):
        n_components=n
    mu=matrix.mean(axis=0)
    X=matrix-mu
    if n>d: # if the number of training set is greater than the size of the image, which is almost impossible
        Mat=np.dot(X.T, X)
        [eigenvalues, eigenvectors]=np.linalg.eigh(Mat)
        # print eigenvalues
    else:  # if the number of training set is less than the size of the image
        Mat=np.dot(X, X.T)
        [eigenvalues, eigenvectors]=np.linalg.eigh(Mat)
        # print eigenvalues
        eigenvectors=np.dot(X.T, eigenvectors)
        for i in range(n):
            eigenvectors[:,i]= eigenvectors[:,i]/np.linalg.norm(eigenvectors[:, i])
    # sort the eigenvalues in descending order
    sortedIndex=np.argsort(-eigenvalues)
    eigenvectors=eigenvectors[:,sortedIndex]
    #select only n_components
    eigenvalues_return=eigenvalues[0:n_components]
    eigenvectors_return=eigenvectors[:,0:n_components]
    return [eigenvalues_return, eigenvectors_return]


## The eigenvectors we have calculated can contain negative values, but the image data is expected as unsigned integer values
## in the range of 0~255
def range_adjust(img, lo, hi, dtype=None):
    X=np.asarray(img)
    Adrange=hi-lo
    if Adrange<=0:
        return np.asarray(img, dtype=dtype)
    min_img, max_img= np.min(img), np.max(img)
    img=img-float(min_img)
    img=img/float(max_img-min_img)
    img=img*Adrange+lo
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def reconstruct(im_target, mu, steps, eigenfaces):
    create_plot(title="image to be reconstructed", img=im_target)
    faces_reconstruct = []
    for i in range(min(len(steps), 16)):
        num_faces = steps[i]
        project_matrix = eigenfaces[:, 0:num_faces]
        img_projected = np.dot(im_target.reshape(1, -1) - mu, project_matrix)
        img_reconstruct = np.dot(img_projected, project_matrix.T) + mu
        img_reconstruct = img_reconstruct.reshape(im_target.shape)
        faces_reconstruct.append(range_adjust(img_reconstruct, 0, 255))
    create_plot_eigenface(title="Process of Reconstruct", eigenfaces=faces_reconstruct, rows=4, cols=4)

    im_fulprojected = np.dot(im_target.reshape(1, -1) - mu, eigenfaces)
    im_fulreconstruct = np.dot(im_fulprojected, eigenfaces.T) + mu
    im_fulreconstruct = range_adjust(im_fulreconstruct.reshape(im_target.shape), 0, 255)
    create_plot(title="image fully reconstructed", img=im_fulreconstruct)

def create_plot_eigenface(title, eigenfaces, rows, cols):
    fig=plt.figure()
    fig.text(.5, .95,title, horizontalalignment='center')
    for i in range(len(eigenfaces)):
        ax0=fig.add_subplot(rows, cols, i+1)
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        plt.title("Eigenfaces: #%d" % (i + 1))
        plt.imshow(np.asarray(eigenfaces[i]), cmap=cm.gray)
        #plt.show()
        fig.savefig("pca_eigenfaces.png")
    plt.show()

def create_plot(title, img):
    fig=plt.figure()
    fig.text(.5, .95,title, horizontalalignment='center')
    axes = plt.subplot(111)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.imshow(np.asarray(img), cmap=cm.gray)
    # plt.setp(ax0.get_xticklabels(), visible=False)
    # plt.setp(ax0.get_yticklabels(), visible=False)
    plt.show()


# steps=[i for i in range(10, min(len(imgSet), 320), 20)]
# E=[]
# Fig=plt.figure()
# plt.imshow(X[0],cmap=cm.gray)
# plt.show()
#
#
# for i in xrange(min(len(steps), 16)):
#     numEvs=steps[i]
#     P=project(W[:, 0:numEvs], X[0].reshape(1,-1), mu)
#     R=reconstruct(W[:,0:numEvs], P, mu)
#     R=R.reshape(X[0].shape)
#     E.append(normalize(R, 0, 255))
# subplot ( title =" Reconstruction AT&T Facedatabase ", images =E, rows =4, cols =4,
#           sptitle ="Eigenvectors ", sptitles =steps , colormap =cm.gray , filename ="python_pca_reconstruction.png")
