import csv

import numpy as np


class DataLoader():
    def load_training_data(self, path):
        try:
            reader = csv.reader(open(path, "r"))
        except:
            return None, None
        x = list(reader)
        result = np.array(x).astype(np.float32)
        xTrain = result[:, 1:]
        yTrain = result[:, :1].astype(np.int32)
        yTrain -= 1
        numOfSamples = len(xTrain.T[0])
        matrix = np.zeros([numOfSamples, 3, 32, 32])
        for i, sample in enumerate(xTrain):
            sample = sample.reshape(3, 32, 32)
            matrix[i] = sample
        return matrix, yTrain

    def load_test_data(self, path):
        reader = csv.reader(open(path, "r"))
        x = list(reader)
        xTest = np.array(x)
        xTest = xTest[:, 1:]
        xTest = xTest.astype(np.float32)
        numOfSamples = len(xTest.T[0])
        matrix = np.zeros([numOfSamples, 3, 32, 32])
        for i, sample in enumerate(xTest):
            sample = sample.reshape(3, 32, 32)
            matrix[i] = sample
        return matrix
