import numpy as np
import mnist
import matplotlib.pyplot as plt

def load_data():
    X_train = np.reshape(mnist.train_images(),(60000,784))
    Y_train = mnist.train_labels()
    X_test = np.reshape(mnist.test_images(),(10000,784))
    Y_test = mnist.test_labels()
    
    
    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test = load_data()
print(X_test.shape,X_train.shape)
print(Y_test.shape,Y_train.shape)
print(Y_test)



