import matplotlib
from lib.libsvm.python.svmutil import *
import scipy.io as sio
from PIL import Image
from scipy.misc import imshow
import numpy as np




'''
a. Partition the provided face data into your training and testing data, in a way you choose.
Explain briefly the way you partitioned. Apply PCA to your training data, by computing the
eigenvectors and eigenvalues of the data covariance matrix S=(1/N)AAT directly. Show
and discuss the results, including: the eigenvectors, the eigenvalues, and the mean image,
how many eigenvectors with non-zero eigenvalues are obtained and how many
eigenvectors are to be used for face recognition. Give insights and reasons behind your
answers.

b. Apply PCA to your training data, using the eigenvectors and eigenvalues of (1/N)ATA.
Show and discuss the results in comparison to the above, including: if the eigenvectors
and eigenvalues obtained are identical, what are the pros/cons of each method. Show
respective measurements for your answers.
'''

face_data = sio.loadmat('/Users/Gustaf/Dropbox KTH/Dropbox/KTH/Imperial College London/kurser/autumn/pattern recognition/cw/PCA and SVM for face recognition/lib/face.mat')
face_labels = face_data['l']
face_vectors = face_data['X']
first_face = face_vectors[:, 420]
len_img = len(first_face)

img = Image.fromarray(first_face.reshape((46, 56)).transpose())
img.show()

#class Eigenfaces(object):

#    def __init__(self):

