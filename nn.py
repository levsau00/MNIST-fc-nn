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
def d_relu(z):
    return z>0

# network definition
class fcnn:
    def __init__(self,learning_rate = 0.05):
        self.layer1 = np.ndarray((10,1))
        self.layer2 = np.ndarray((10,1))
        self.w1 = np.random.rand(10,784) - 0.5
        self.w2 = np.random.rand(10,10) - 0.5
        self.b1 = np.random.rand(10) - 0.5
        self.b2 = np.random.rand(10) - 0.5
        self.z1 = np.ndarray((10,1))
        self.z2 = np.ndarray((10,1))
        self.learning_rate = learning_rate

    def forward(self,X):
        self.z1 = self.w1.dot(X) + self.b1
        self.layer1 = relu(self.z1)

        self.z2 = self.w2.dot(self.layer1) + self.b2
        self.layer2 = softmax(self.z2)
        print("z1:",self.z1.shape,"\nz2:",self.z2.shape)
        
        return self.layer2

    def backward(self, X, Y):
        dz2 = self.layer2 - to_onehot(Y)
        dw2 = dz2.dot(self.layer1.T)
        db2 = dz2
        dz1 = self.w2.dot(dz2) * d_relu(self.z1)
        dw1 = np.reshape(dz1,(dz1.shape[0],1)).dot(np.reshape(X, (1,X.shape[0])))
        db1 = dz1

        return dw1,db1,dw2,db2
    
    def update_params(self,dw1,db1,dw2,db2):
        self.w1 -= self.learning_rate*dw1
        self.b1 -= self.learning_rate*db1
        self.w2 -= self.learning_rate*dw2
        self.b2 -= self.learning_rate*db2
