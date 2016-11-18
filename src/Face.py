import numpy as np


class Face(object):

    def __init__(self, label, face_vector):
        self.label = label
        self.face_vector = face_vector.astype(float)

    def get_label(self):
        return self.label

    def get_face_vector(self):
        return self.face_vector

    def get_dimension(self):
        return self.face_vector.shape[0]
