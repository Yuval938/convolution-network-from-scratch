import numpy as np

from NN.utils import utils


class CNN_Model():
    def __init__(self, layers: [], loss, optimizer: str):
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss

    def forward(self, x, y):
        for layer in self.layers:
            x = layer.forward(x)
            # print(x.shape)
        y_hat = x  # just for readability
        one_hot_y = utils.hot_reloading(y, 10)
        loss = self.loss.calc(one_hot_y, y_hat)
        self.backward(one_hot_y)
        return int(np.argmax(y_hat)),loss

    def backward(self, y):
        cache = y
        for layer in list(reversed(self.layers)):
            cache = layer.backward(cache)

    def step(self, batchsize, eta):
        for layer in list(reversed(self.layers)):
            layer.step(batchsize=batchsize,eta=eta)

    def predict(self, x, y):
        for layer in self.layers:
            x = layer.forward(x)
        y_hat = x  # just for readability
        one_hot_y = utils.hot_reloading(y, 10)
        loss = self.loss.calc(one_hot_y, y_hat)
        return int(np.argmax(y_hat)),loss
    def train(self,epochs,batchsize,eta,xTrain,yTrain,xValidate=None,yValidate=None):
        print("starting training...")
        for i in range(1, epochs):
            print(f"epoch number {i}:")
            Vcounter = 0
            counter = 0
            sample1 = 0
            totalloss = 0
            Vtotalloss = 0
            for sample, y in zip(xTrain, yTrain):
                sample1 += 1
                y_hat, loss = self.forward(sample, int(y[0]))
                totalloss += loss
                if int(y[0]) == y_hat:
                    counter += 1
                if sample1 == 1 or sample1 % 50 == 0:
                    # if sample1%1000 ==0:
                    #     #print(sample1)
                    self.step(batchsize, eta)
            print(f"training,acc: {(counter / len(yTrain)) * 100}%, loss: {(totalloss / len(yTrain)) * 100}")
            if xValidate is not None:
                for sample, y in zip(xValidate, yValidate):
                    y_hat, Vloss = self.predict(sample, int(y[0]))
                    Vtotalloss += Vloss
                    if int(y[0]) == y_hat:
                        Vcounter += 1
                print(f"validation,acc: {(Vcounter / len(yValidate)) * 100}%, loss: {(Vtotalloss / len(yValidate)) * 100}")

    def test(self, xTest):
        predictions = []
        for sample in zip(xTest):
            predictions.append(self.predict_test(sample[0]))
        with open("output.txt", "w+") as pred:
            pred.write('\n'.join(str(v) for v in predictions))

        pass
    def predict_test(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        y_hat = x  # just for readability
        return int(np.argmax(y_hat))
