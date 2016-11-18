
class EigenObject(object):

    def __init__(self, eigenvalue, eigenvector):
        self.eigenvalue = eigenvalue
        self.eigenvector = eigenvector

    def get_eigenvalue(self):
        return self.eigenvalue

    def get_eigenvector(self):
        return self.eigenvector