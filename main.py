import numpy as np
import matplotlib.pyplot as plt
import gzip

files = ["MNIST\mnist-train-images.gz","MNIST\mnist-train-labels.gz","MNIST\mnist-test-images.gz","MNIST\mnist-test-labels.gz"]


def load_data():
    with gzip.GzipFile(files[0]) as f:
        X_train = np.fromfile(f)
    with gzip.GzipFile(files[1]) as f:
        Y_train = np.fromfile(f)
    with gzip.GzipFile(files[2]) as f:
        X_test = np.fromfile(f)
    with gzip.GzipFile(files[3]) as f:
        Y_test = np.fromfile(f)
    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test = load_data()
print(X_test.shape,X_train.shape)