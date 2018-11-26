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
        trn_data_keep = trn_data[0,:].reshape(1,len(trn_data[0,:]))
        labels_keep = np.array(labels[0]).reshape(1)
        for j in range(1,trn_data.shape[0]):
            # Step 1 IB1 algorithm------------
            # sim_y = np.argpartition([self.d(trn_data[j, :], trn_sample) for trn_sample in trn_data_keep],kth=self.k)[:self.k]

            # Step 2 IB1 algorithm------------
            # y_max = labels[sim_y]

            # Step 3 IB1 algorithm: Not implemented since it is not useful

            # Step 4 IB1 algorithm------------
            trn_data_concat = trn_data[j,:].reshape(1,len(trn_data[j,:]))
            trn_data_keep = np.concatenate((trn_data_keep,trn_data_concat),axis=0)
            labels_keep = np.concatenate((labels_keep, np.array(labels[j]).reshape(1)))
        self.trn_data = trn_data_keep
        self.trn_labels = labels_keep



    def classify(self, tst_data):
        self.tst_labels = np.zeros((tst_data.shape[0], 1))
        for i in range(tst_data.shape[0]):
            neighbor_idxs = np.argpartition([self.d(tst_data[i,:], trn_sample) for trn_sample in self.trn_data],
                                            kth=self.k)[:self.k]

            # neighbor_idxs = np.argsort([self.d(tst_data[i,:], trn_sample) for trn_sample in self.trn_data])[:self.k]
            # labels, counts = np.unique(self.trn_labels[neighbor_idxs], return_counts=True)
            # self.tst_labels[i] = labels[np.argmax(counts)]
            self.tst_labels[i] = self.trn_labels[neighbor_idxs]