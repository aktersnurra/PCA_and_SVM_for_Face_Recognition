from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

from ReadFaces import ReadFaces
from EigenObject import EigenObject


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


class EigenFaces(object):

    def __init__(self):
        self.face_data = ReadFaces()
        self.labeled_faces, self.training_data, self.test_data = self.face_data()
        self.D = self.labeled_faces[0].get_dimension()
        self.x_bar = np.zeros((self.D, 1)).astype(float)
        self.N = len(self.training_data)
        self.S = np.zeros((self.D, self.D)).astype(float)
        self.A = np.zeros((self.D, self.N)).astype(float)
        self.eig_arr = []
        self.eig_values = []
        self.M = 200
        self.w = []

    def __call__(self):
        print 'Computing...'
        self.compute_avg_face_vector()
        self.compute_covariance_matrix()
        self.compute_eigenvectors()
        self.display_eigenvalues()
        self.faces_onto_eigenfaces()
        self.reconstruct_face()

    def display_face(self, face_vector):
        img = face_vector.reshape((46, 56)).transpose()
        plt.imshow(img, cmap='Greys_r')
        plt.show()

    def compute_avg_face_vector(self):
        '''
        compute the average face vector

        x_bar = (1/N)sum_{n=1}^N

        '''
        x_sum = np.zeros((self.D, 1)).astype(float)
        for face in self.training_data:
            x = face.get_face_vector()
            x_sum += x
        self.x_bar = (1 / self.N) * x_sum

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
        for face in self.training_data:
            x = face.get_face_vector()
            self.A[:, counter] = x[:, 0] - self.x_bar[:, 0]
            counter += 1
        self.S = np.cov(self.A)

    def compute_eigenvectors(self):
        '''
        Computes the eigenvalues and eigenvectors for the training data. Zips together eigenvalues with corresponding eigenvector.
        Sorts the zip list on eigenvalues form high to low.
        '''
        eig_values, eig_matrix = LA.eig(self.S)
        start = 0
        stop = self.D
        for i in xrange(start, stop):
            self.eig_values.append(eig_values[i].real)
            self.eig_arr.append(EigenObject(eig_values[i].real, eig_matrix[:, i].real))
        self.display_face(self.eig_arr[0].get_eigenvector())

    def display_eigenvalues(self):
        '''
        Plots eigenvalues vs dimension of the images.
        '''
        plt.plot(np.array(np.arange(self.D)), self.eig_values[:])
        plt.ylim([self.eig_values[-1], self.eig_values[0]])
        plt.xlim([0, self.D])
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalue')
        plt.show()

    def faces_onto_eigenfaces(self):
        '''
        Representing faces onto eigenfaces

        w_n = [a_n1, a_n2, ..., a_nM].t

        where

        a_ni = phi_n^T*u_i, i = 1, ..., M

        '''
        phi = np.zeros((self.D, 1)).astype(float)
        for k in xrange(0, self.N):
            phi[:, 0] = self.A[:, k]
            w_n = np.zeros(self.M)
            for i in xrange(0, self.M):
                u = self.eig_arr[i].get_eigenvector()
                phi_transpose = np.transpose(phi)
                a_ni = phi_transpose.dot(u)
                w_n[i] = a_ni
            self.w.append(w_n)

    def reconstruct_face(self):
        sum_eig_vectors = np.zeros((self.D, 1)).astype(float)
        x_tilde = np.zeros((self.D, 1)).astype(float)
        u = np.zeros((self.D, 1))
        for i in xrange(0, self.M):
            u[:, 0] = self.eig_arr[i].get_eigenvector()
            a_ni = self.w[0][i]
            sum_eig_vectors += a_ni * u
        x_tilde[:, 0] = self.x_bar[:, 0] + sum_eig_vectors[:, 0]
        self.display_face(x_tilde)

run = EigenFaces()
run()
