from algorithms.distances import euclidean
import numpy as np


class ib1Algorithm():

    trn_data = None
    trn_labels = None
    tst_labels = None

    def __init__(self, k=1, metric='euclidean'):
        self.k = k
        if metric == 'euclidean':
            self.d = euclidean

    def fit(self, trn_data, labels):
        self.trn_data = trn_data
        self.trn_labels = labels

    def classify(self, tst_data):
        self.tst_labels = np.zeros((tst_data.shape[0], 1))
        for i in range(tst_data.shape[0]):
            neighbor_idxs = np.argpartition([self.d(tst_data[1,:], trn_sample) for trn_sample in self.trn_data], kth=self.k)[:self.k]
            labels, counts = np.unique(self.trn_labels[neighbor_idxs], return_counts=True)
            self.tst_labels[i] = labels[np.argmax(counts)]