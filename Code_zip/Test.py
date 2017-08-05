
#################################################################
###############Test for correctness##############################
#################################################################
import face_util_func
from face_util_func import range_adjust, create_plot, reconstruct
from face_util_func import read_images, pca, asMatrix, create_plot_eigenface
import os, sys
from PIL import Image
import numpy as np
import copy
##1.Compute the principal components using first 190 people's neutral expression image

path = "C:\Users\\12815\\Desktop\\face\\Training Set"
#[imgSet, y] = read_images(path)
# Question 8
[imgSet_origin, y] = read_images(path)
sigma=50
imgSet=copy.deepcopy(imgSet_origin)
imgSet[0]+=sigma*np.random.randn(193, 162).astype(np.uint8)
imgMatrix=asMatrix(imgSet)
n_components=len(imgMatrix)
[eigenvalues, eigenfaces] = pca(imgMatrix,0)
# print eigenvalues
Eigen_face_show=[]
for i in range(min(len(imgSet), 16)):
    e=eigenfaces[:,i].reshape(imgSet[0].shape)
    Eigen_face_show.append(range_adjust(e,0,255))
create_plot_eigenface(title =" Eigenfaces of face", eigenfaces=Eigen_face_show, rows =4, cols =4)
mu=imgMatrix.mean(axis=0)
steps=[20, 30, 50, 60, 70, 80, 90, 100, 120, 140, 150, 160, 180, 190]

# # 2.Resconstruct one of the 190 people's neutral expression image Using different number of eigenfaces

# path = "C:\Users\\12815\\Desktop\\face\\Training Set"
# im_target=Image.open(path+'/'+'39a.jpg')
# im_target=im_target.convert("L")
# im_target=np.asarray(im_target, dtype=np.uint8)
# reconstruct(im_target, mu, steps, eigenfaces)

#
# # 3.reconstruct one of 190 people's smiling expression using different number of PCs
# path = "C:\Users\\12815\\Desktop\\face"
# im_target=Image.open(path+'/'+'39b.jpg')
# # im_target=im_target.convert("L")
# im_target=np.asarray(im_target, dtype=np.uint8)
# reconstruct(im_target, mu, steps, eigenfaces)

# # 4.reconstruct one of the other people's neutral expression image using different number of PCs
# path = "C:\Users\\12815\\Desktop\\face\\Test Set"
# im_target=Image.open(path+'/'+'193a.jpg')
# im_target=im_target.convert("L")
# im_target=np.asarray(im_target, dtype=np.uint8)
# reconstruct(im_target, mu, steps, eigenfaces)

# #5.Use other non-human image to reconstruct it using all PCS
#
# path = "C:\Users\\12815\\Desktop\\face\\Not_face"
# im_target=Image.open(path+'/'+'Horse.jpg')
# im_target=im_target.convert("L")
# size=(162, 193)
# im_target=im_target.resize(size,Image.ANTIALIAS)
# im_target=np.asarray(im_target, dtype=np.uint8)
# create_plot(title="image to be reconstructed",img=im_target)
# im_fulprojected=np.dot(im_target.reshape(1,-1)-mu, eigenfaces)
# im_fulreconstruct=np.dot(im_fulprojected, eigenfaces.T)+mu
# im_fulreconstruct=range_adjust(im_fulreconstruct.reshape(im_target.shape),0,255)
# create_plot(title="image fully reconstructed",img=im_fulreconstruct)
#
# # # 6.  Rotate the image and Observe the difference
# rotate_degree=[5, 15, 25, 35, 45]
# im_target_origin=Image.open(path+'/'+'39a.jpg')
# for degree in rotate_degree:
#     im_target=im_target_origin.rotate(degree)
#     im_target = im_target.convert("L")
#     size = (162, 193)
#     im_target = im_target.resize(size, Image.ANTIALIAS)
#     im_target = np.asarray(im_target, dtype=np.uint8)
#     title_temp="image to be reconstructed, rotated degree="+str(degree)
#     create_plot(title=title_temp, img=im_target)
#     im_fulprojected = np.dot(im_target.reshape(1, -1) - mu, eigenfaces)
#     im_fulreconstruct = np.dot(im_fulprojected, eigenfaces.T) + mu
#     im_fulreconstruct = range_adjust(im_fulreconstruct.reshape(im_target.shape), 0, 255)
#     create_plot(title="Rotated image fully reconstructed", img=im_fulreconstruct)

# # 7. Adding noise to one of the 10 people's neutral expression image and reconstruct
# path = "C:\Users\\12815\\Desktop\\face\\Test Set"
# im_target_origin=Image.open(path+'/'+'191a.jpg')
# im_target_origin=im_target_origin.convert("L")
# im_target_origin=np.asarray(im_target_origin, dtype=np.uint8)
# sigma_noise=[1,10,40,80]
# for sigma in sigma_noise:
#     noise=sigma*np.random.randn(193, 162)
#     im_target=noise+im_target_origin
#     reconstruct(im_target, mu, steps, eigenfaces)

#
#
#
# 8. Reconstruct with contaminated pic
im_target=imgSet[0]
reconstruct(im_target, mu, steps, eigenfaces)



