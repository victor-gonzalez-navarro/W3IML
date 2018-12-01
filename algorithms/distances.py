import numpy as np

#def euclidean(a, b):
#    return np.sum((a-b)**2)

#def euclidean2(a, b):
#    return np.sqrt(np.sum((a-b)**2))


def euclidean(a, b):
    distance = 0
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance = distance + (ai-bi)**2
        else:
            if ai != bi:
                distance = distance + 1
    return distance


def manhattan(a, b):
    distance = 0
    for ai, bi in zip(a,b):
        if type(ai) in [float, np.float64]:
            distance = distance + abs(ai-bi)
        else:
            if ai != bi:
                distance = distance + 1
    return distance
