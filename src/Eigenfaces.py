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
        self.face_data = ReadFaces()                                                #Object of class ReadFaces
        self.labeled_faces, self.training_data, self.test_data = self.face_data()   #gets the usefull data from ReadFaces object
        self.D = self.labeled_faces[0].get_dimension()                              #Set the dimension of the face vectors
        self.x_bar = np.zeros((self.D, 1)).astype(float)
        self.x_tilde = np.zeros((self.D, 1)).astype(float)
        self.N = len(self.training_data)                                            #Sets number of faces in the used for training
        self.S = np.zeros((self.D, self.D)).astype(float)
        self.A = np.zeros((self.D, self.N)).astype(float)
        self.eig_arr = []
        self.eig_values = []
        self.M = 300                                                                #How many eigenvectors used for face reconstruction
        self.weight_matrix = []                                                     #Contains each weight array used for face reconstruction
        self.face = 0                                                               #Detirmines the face that will be reconstructed
        self.arr_of_subspaces = [50, 100, 150, 200, 250, 289]
        self.error_arr = []
        self.path_to_lib = '/Users/Gustaf/Dropbox KTH/Dropbox/KTH/Imperial College London/kurser/' \
                           'autumn/pattern recognition/cw/PCA and SVM for face recognition/lib/'

    def __call__(self):
        '''
        runs the methods it the class.
        :return:
        '''
        print 'Computing...'
        self.compute_avg_face_vector()
        self.compute_covariance_matrix()
        self.compute_eigenvectors()
        #self.display_eigenvalues()
        self.faces_onto_eigenfaces()
        #self.reconstruct_face()
        #self.save_faces()
        self.compute_error()
        self.display_error()
        #self.compute_zero_eigenvalues()

    def compute_avg_face_vector(self):
        '''
        compute the average face vector:
        x_bar = (1/N)sum_{n=1}^N
        '''
        x_sum = np.zeros((self.D, 1)).astype(float)
        for face in self.training_data:
            x = face.get_face_vector()
            x_sum += x
        self.x_bar = (1 / self.N) * x_sum

    def compute_covariance_matrix(self):
        '''
        compute the covariance matrix S:
        S = 1 / N AAT

        where:
        phi_n = x_n - x_bar

        and:
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

    def faces_onto_eigenfaces(self):
        '''
        Representing faces onto eigenfaces:
        w_n = [a_n1, a_n2, ..., a_nM].t

        where:
        a_ni = phi_n^T*u_i, i = 1, ..., M
        '''
        phi = np.zeros((self.D, 1)).astype(float)
        for k in xrange(0, self.N):
            phi[:, 0] = self.A[:, k]
            weight_array = np.zeros(self.M)
            for i in xrange(0, self.M):
                u = self.eig_arr[i].get_eigenvector()
                phi_transpose = np.transpose(phi)
                weight = phi_transpose.dot(u)
                weight_array[i] = weight
            self.weight_matrix.append(weight_array)

    def reconstruct_face(self):
        sum_eig_vectors = np.zeros((self.D, 1)).astype(float)
        u = np.zeros((self.D, 1))
        for i in xrange(0, self.M):
            u[:, 0] = self.eig_arr[i].get_eigenvector()
            a_ni = self.weight_matrix[self.face][i]
            sum_eig_vectors += a_ni * u
        self.x_tilde[:, 0] = self.x_bar[:, 0] + sum_eig_vectors[:, 0]

    def display_face(self, face_vector, title):
        img = face_vector.reshape((46, 56)).transpose()
        plt.figure(figsize=(0.48, 0.58))
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
        plt.savefig(self.path_to_lib + title)
        plt.show()

    def compute_zero_eigenvalues(self):
        threshold = 1e-10
        counter = 0
        eig_zero_arr = []
        for eigenvalue in self.eig_values:
            if np.absolute(eigenvalue) < threshold:
                eig_zero_arr.append(eigenvalue)
                counter += 1
        print counter, eig_zero_arr

    def compute_error(self):
        original_face = np.copy(self.training_data[self.face].get_face_vector())
        arr_of_rec_face = []

        for subspace in self.arr_of_subspaces:
            self.M = subspace
            self.reconstruct_face()
            arr_of_rec_face.append(np.copy(self.x_tilde))

        for rec_face in arr_of_rec_face:
            self.error_arr.append(LA.norm(original_face - rec_face))

        print 'error', self.error_arr

    def display_error(self):
        title = self.path_to_lib + '/rec_error.pdf'
        plt.stem(self.arr_of_subspaces[:], self.error_arr[:])
        plt.ylim([self.error_arr[-1], self.error_arr[0]+16])
        plt.xlim([self.arr_of_subspaces[0]-10, self.arr_of_subspaces[-1]+10])
        plt.xlabel('Subspace dimension')
        plt.ylabel('Error')
        plt.savefig(title)
        plt.show()

    def save_faces(self):
        '''
        Save each face to /lib as .pdf.
        :return:
        '''
        self.display_face(self.x_bar, '/avg_face_vector.pdf')

        for i in range(0, 5):
            title = '/eig_face_vector' + str(i) + '.pdf'
            self.display_face(self.eig_arr[i].get_eigenvector(), title)

        self.display_face(self.x_tilde, 'reconstructed_face' + str(self.face) + '.pdf')

    def display_eigenvalues(self):
        '''
        Plots eigenvalues vs dimension of the images.
        '''
        title = self.path_to_lib + '/eigvalues_vs_dim_500zoom.pdf'
        plt.ticklabel_format(style='sci', axis='y', scilimits=(1, 3))
        plt.plot(np.array(np.arange(self.D)), self.eig_values[:])
        plt.ylim([self.eig_values[-1], self.eig_values[0]])
        plt.xlim([0, 250])
        plt.xlabel('Dimension')
        plt.ylabel('Eigenvalue')
        plt.savefig(title)
        plt.show()

run = EigenFaces()
run()
