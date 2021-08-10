import numpy as np

from NN.utils import utils


class NLL():
    def calc(self,y,y_hat):
        return np.sum(-y * np.log(y_hat))
