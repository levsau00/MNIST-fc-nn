import numpy as np

# activation functions
def relu(z): 
    z = np.maximum(z,0)
    return z
def softmax(z):    
    num = np.exp(z-np.max(z))
    denom = np.sum(num)
    return num/denom 
def to_onehot(label):
    onehot = np.zeros(10)
    onehot[label] = 1
    return onehot
    

# network definition
class fcnn:
    def __init__(self):
        self.layer1 = np.ndarray((10,1))
        self.layer2 = np.ndarray((10,1))
        self.w1 = np.random.rand(10,784) - 0.5
        self.w2 = np.random.rand(10,10) - 0.5
        self.b1 = np.random.rand(10) - 0.5
        self.b2 = np.random.rand(10) - 0.5

    

    def forward(self,X):
        z1 = self.w1.dot(X) + self.b1
        self.layer1 = relu(z1)

        z2 = self.w2.dot(self.layer1) + self.b2
        self.layer2 = softmax(z2)
        print("z1:",z1.shape,"\nz2:",z2.shape)
        
        return self.layer2

    def backward(self, Y):
        dz2 = self.layer2 - Y
        dw2 = dz2.dot(self.layer1.T)
        db2 = dz2
        da1 = self.w2.dot(dz2)

        return dw1,db1,dw2,db2