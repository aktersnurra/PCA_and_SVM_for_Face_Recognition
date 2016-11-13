from __future__ import division
import matplotlib
from lib.libsvm.python.svmutil import *
import scipy.io as sio
from PIL import Image
from scipy.misc import imshow
import numpy as np
from FaceData import FaceData


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

'''face_data = sio.loadmat(path_to_faces)
face_labels = face_data['l']
face_vectors = face_data['X']

for i in range(100,110):
    img = Image.fromarray(face_vectors[:, i].reshape((46, 56)).transpose())
    img.show()'''


class EigenFaces(object):

    def __init__(self, face_data):
        self.face_data = face_data
        self.labeled_faces, self.training_data, self.test_data = self.face_data()
        self.x_bar = np.zeros(len(self.training_data[0][1]))
        self.N = len(self.training_data)
        self.D = len(self.training_data[0][1])

    def printdata(self):
        print str(len(self.training_data)) + ' ' + str(len(self.test_data))
        print self.test_data

    def compute_avg_face_vector(self):
        '''
        compute the average face vector

        x_bar = (1/N)sum_{n=1}^N

        '''
        x_sum = np.zeros(self.D)
        for face in self.training_data:
            x = face[1]
            x_sum += x
        self.x_bar = (1 / self.N) * x_sum
        img = Image.fromarray(self.x_bar.reshape((46, 56)).transpose())
        img.show()

    def compute_covariance_matrix(self):
        '''

        compute the covariance matrix S

        S = 1 / N AAT
        where

        phi_n = x_n - x_bar

        and

        A = [phi_1, ... , phi_n]

        '''
        counter = 0
        A = np.zeros((self.D,self.N))
        for face in self.training_data:
            x = face[1]
            phi = x - self.x_bar
            A[:,counter] = phi
            counter += 1

        AT = A.transpose()
        S = 1 / self.N * np.dot(A, AT)
        print S.shape




face_data = FaceData()
eigenfaces = EigenFaces(face_data)
#eigenfaces.printdata()
eigenfaces.compute_avg_face_vector()
eigenfaces.compute_covariance_matrix()