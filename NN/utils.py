import numpy as np


class utils():
    @classmethod
    def hot_reloading(self,y, size):
        y_hot_realoded = np.zeros((size, 1))
        y_hot_realoded[int(y)] = 1
        return y_hot_realoded.T
