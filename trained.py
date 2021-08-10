import pickle
import sys

from DataLoader import DataLoader

if __name__ == '__main__':
    if len(sys.argv) == 1:
        filename = 'trained_model.sav'
        testcsv = "test.csv"
    else:
        filename=sys.argv[1]
        testcsv = sys.argv[2]
    dataloader =DataLoader()
    xTest = dataloader.load_test_data(testcsv)
    loaded_model = pickle.load(open(filename, 'rb'))
    loaded_model.test(xTest)