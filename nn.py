import numpy as np

class fcnn:
    def __init__(self):
        self.layer1 = np.ndarray((784,1))
        self.layer2 = np.ndarray((10,1))
        self.layer3 = np.ndarray((10,1))
        self.w1 = np.ndarray((10,784))
        self.w2 = np.ndarray((10,10))
        self.b1 = np.ndarray((10,1))
        self.b2 = np.ndarray((10,1))

        