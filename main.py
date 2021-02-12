import numpy as np
import mnist
import matplotlib.pyplot as plt
from nn import fcnn

def load_data():
    X_train = np.reshape(mnist.train_images(),(60000,784))
    Y_train = np.reshape(mnist.train_labels(),(60000))
    X_test = np.reshape(mnist.test_images(),(10000,784))
    Y_test = np.reshape(mnist.test_labels(),(10000))
    
    
    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test = load_data()
#print(X_test.shape,X_train.shape)
#print(Y_test.shape,Y_train.shape)

net = fcnn()
print("forward output:",net.forward(X_train[10]))
print(net.backward(X_train[10],Y_train[10]).shape)
#print(X_test[0].shape)


