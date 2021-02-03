import numpy as np

class fcnn:
    def __init__(self):
        self.layer1 = np.ndarray((1,784))
        self.layer2 = np.ndarray((1,784))
        self.layer3 = np.ndarray((1,10))
        self.weight1= np.ndarray((1,784))
        self.weight2= np.ndarray((1,784))