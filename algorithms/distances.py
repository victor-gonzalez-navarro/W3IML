import numpy as np

#def euclidean(a, b):
#    return np.sum((a-b)**2)

#def euclidean2(a, b):
#    return np.sqrt(np.sum((a-b)**2))

def euclidean(a, b):
    distance = 0
    for ai,bi in zip(a,b):
        distance = distance + (ai-bi)**2
    return distance

def manhattan(a, b):
    distance = 0
    for ai,bi in zip(a,b):
        distance = distance + abs(ai-bi)
    return distance