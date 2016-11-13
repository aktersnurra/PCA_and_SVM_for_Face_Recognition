import scipy.io as sio


class FaceData(object):

    def __init__(self):
        self.path_to_faces = '/Users/Gustaf/Dropbox KTH/Dropbox/KTH/Imperial College London/kurser/autumn/pattern recognition/cw/PCA and SVM for face recognition/lib/face.mat'
        self.face_data = sio.loadmat(self.path_to_faces)
        self.face_labels = self.face_data['l']
        self.face_vectors = self.face_data['X']
        self.labeled_faces = []
        self.training_data = []
        self.test_data = []

    def __call__(self):

        self.label_vectors()
        self.partition_data()

        return self.labeled_faces, self.training_data, self.test_data

    def label_vectors(self):
        '''Create a list with tuples containing label and corresponding face vector.'''
        start = 0
        stop = len(self.face_labels[0]) - 1
        for i in range(start, stop):
            face_label = self.face_labels[0][i]
            face_vector = self.face_vectors[:, i]
            self.labeled_faces.append((face_label, face_vector))

    def partition_data(self):
        '''Partition data into training and testing data. The data is divided 60/40.'''
        check = range(1, 7)
        counter = 1
        start = 0
        stop = len(self.labeled_faces) - 1
        for i in range(start, stop):
            if counter in check:
                self.training_data.append(self.labeled_faces[i])
                counter += 1
            else:
                self.test_data.append(self.labeled_faces[i])
                counter += 1

            if counter == 10:
                counter = 1






