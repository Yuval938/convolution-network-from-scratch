import pickle

from DataLoader import DataLoader
from NN.CNN_MODEL import CNN_Model
from NN.Layers import Conv, Flatten, Linear, LinearRelu, Softmax, Dropout, MaxPool, TanH
from NN.NLL import NLL

if __name__ == '__main__':
    dataloader = DataLoader()
    xTrain, yTrain = dataloader.load_training_data("./train.csv")
    xValidate, yValidate = dataloader.load_training_data("./validate.csv")
    conv = Conv(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
    conv2 = Conv(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    conv3 = Conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
    fc1 = Linear(in_size=1024, out_size=256, uniform_range=0.1)
    fc2 = Linear(in_size=256, out_size=10, uniform_range=0.1)
    model = CNN_Model(
        layers=[conv, TanH(), MaxPool(filter_size=2), conv2, TanH(), MaxPool(filter_size=2), conv3, TanH(),
                MaxPool(filter_size=2), Flatten(), Dropout(p=0.5), fc1, LinearRelu(), fc2, Softmax()], loss=NLL(),
        optimizer="none")
    if xValidate is not None:
        model.train(epochs=57, batchsize=100, eta=0.035, xTrain=xTrain, yTrain=yTrain, xValidate=xValidate,
                    yValidate=yValidate)
    else:
        model.train(epochs=57, batchsize=100, eta=0.035, xTrain=xTrain, yTrain=yTrain)
    filename = 'trained_model_new.sav'  # this will overwrite existing model!
    pickle.dump(model, open(filename, 'wb'))  # save model

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
