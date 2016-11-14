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
        self.eig_val_sorted = []
        self.eig_zip = []
        self.M = 350
        self.A = np.zeros((self.D, self.N))
        self.w = []
        self.eig_vectors = []

    def display_face(self, face_vector):
        img = Image.fromarray(face_vector.reshape((46, 56)).transpose())
        img.show()

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
        #self.display_face(self.x_bar)

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
            x = face[1]
            phi = x - self.x_bar
            self.A[:, counter] = phi
            counter += 1

        AT = self.A.transpose()
        self.S = 1 / self.N * np.dot(self.A, AT)

    def compute_eigenvectors(self):
        '''
        Computes the eigenvalues and eigenvectors for the training data. Zips together eigenvalues with corresponding eigenvector.
        Sorts the zip list on eigenvalues form high to low.
        '''
        eig_values, eig_vectors = LA.eig(self.S)
        real_eig_values = eig_values.real
        self.eig_zip = zip(real_eig_values, eig_vectors)
        self.eig_zip.sort(key=lambda t: t[0], reverse=True)

    def display_eigenvalues(self):
        '''
        Plots eigenvalues vs dimension of the images.
        '''
        self.eig_val_sorted = [eig_val[0] for eig_val in self.eig_zip]
        plt.plot(np.array(np.arange(self.D)), self.eig_val_sorted[:])
        plt.ylim([self.eig_val_sorted[-1], self.eig_val_sorted[0]])
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
        self.eig_vectors = [eig_vec[1].real for eig_vec in self.eig_zip]

        for k in xrange(0, self.N):
            phi = self.A[:, k]
            w_n = np.zeros(self.M)
            for i in xrange(0, self.M):
                u = self.eig_vectors[i]
                a_ni = phi.transpose().dot(u)
                w_n[i] = a_ni.real
            self.w.append(w_n)

    def reconstruct_face(self):
        eig_vector = np.zeros(self.D)
        for i in xrange(0, self.M):
            u = self.eig_vectors[i].real
            a_ni = self.w[0][i]
            eig_vector += a_ni*np.array(u)

        x_tilde = self.x_bar + eig_vector
        self.display_face(x_tilde)





def main():
    eigenfaces = EigenFaces()
    eigenfaces.compute_avg_face_vector()
    eigenfaces.compute_covariance_matrix()
    eigenfaces.compute_eigenvectors()
    #eigenfaces.display_eigenvalues()
    eigenfaces.faces_onto_eigenfaces()
    eigenfaces.reconstruct_face()

main()