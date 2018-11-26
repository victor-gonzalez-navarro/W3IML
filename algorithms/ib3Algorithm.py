from algorithms.distances import euclidean
import numpy as np


class ib3Algorithm():

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
        classi_list = [[0]]
        for j in range(1,trn_data.shape[0]):
            neighbor = np.argpartition([self.d(trn_data[j,:], trn_sample) for trn_sample in trn_data_keep], kth=0)[:1]
            if labels[j] != labels_keep[neighbor]:
                trn_data_concat = trn_data[j,:].reshape(1,len(trn_data[j,:]))
                trn_data_keep = np.concatenate((trn_data_keep,trn_data_concat),axis=0)
                labels_keep = np.concatenate((labels_keep, np.array(labels[j]).reshape(1)))
                classi_list.append([0])
            if len(labels_keep) > 1:
                for m in range(trn_data_keep.shape[0]):
                    valid_idxs = [x for x in range(trn_data_keep.shape[0]) if x !=m]
                    distances = [self.d(trn_data_keep[m, :], samp) for samp in trn_data_keep[valid_idxs,:]]
                    neighbor = np.argpartition(distances, kth=0)[:1]
                    classi_list[m].append(int(labels_keep[m] == labels_keep[neighbor]))

        remove_idxs = []
        for m in range(trn_data_keep.shape[0]):
            if np.mean(classi_list[m]) < 0.25:
                remove_idxs.append(m)
        trn_data_keep = np.delete(trn_data_keep, remove_idxs, axis=0)
        labels_keep = np.delete(labels_keep, remove_idxs, axis=0)

        self.trn_data = trn_data_keep
        self.trn_labels = labels_keep



    def classify(self, tst_data):
        self.tst_labels = np.zeros((tst_data.shape[0], 1))
        for i in range(tst_data.shape[0]):
            a = tst_data[i,:]
            neighbor_idxs = np.argpartition([self.d(tst_data[i,:], trn_sample) for trn_sample in self.trn_data],
                                            kth=self.k-1)[:self.k]
            labels, counts = np.unique(self.trn_labels[neighbor_idxs], return_counts=True)
            self.tst_labels[i] = labels[np.argmax(counts)]