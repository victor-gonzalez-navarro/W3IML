import numpy as np

def euclidean(a, b):
    return np.sum((a-b)**2)

def euclidean2(a, b):
    return np.sqrt(np.sum((a-b)**2))