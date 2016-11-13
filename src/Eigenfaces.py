from __future__ import division
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

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

    def __init__(self):
        self.face_data = FaceData()
        self.labeled_faces, self.training_data, self.test_data = self.face_data()
        self.x_bar = np.zeros(len(self.training_data[0][1]))
        self.N = len(self.training_data)
        self.D = len(self.training_data[0][1])
        self.S = []

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
        self.S = 1 / self.N * np.dot(A, AT)
        #print self.S.shape

    def compute_eigenvectors(self):
        eig_values, eig_vectors = LA.eig(self.S)
        M = np.array(np.arange(self.D))

        real_eig_val = eig_values.real
        real_eig_sorted = np.array(sorted(real_eig_val, reverse=True))

        plt.plot(M, real_eig_sorted)
        plt.ylim([0,real_eig_sorted[0]])
        plt.xlim([0,self.D])
        plt.show()



def main():
    eigenfaces = EigenFaces()
    eigenfaces.compute_avg_face_vector()
    eigenfaces.compute_covariance_matrix()
    eigenfaces.compute_eigenvectors()

main()