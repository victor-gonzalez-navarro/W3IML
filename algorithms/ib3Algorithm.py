from algorithms.distances import euclidean
import numpy as np


class ib3Algorithm():

    trn_data = None
    trn_labels = None
    tst_labels = None
    z = 1.645   # z90 = 1.645

    def __init__(self, k=1, metric='euclidean'):
        self.k = k
        if metric == 'euclidean':
            self.d = euclidean


    def get_accepted_ys(self, labels_count, incorrect_labels, accuracies, num_processed):
        accepted_idx = []

        for index in range(len(incorrect_labels)):
            accuracy = accuracies[index]
            label_freq = labels_count[incorrect_labels[index]] / num_processed

            limit_conf_accu = np.sqrt(accuracy * (1 - accuracy) / 100) * self.z
            limit_conf_freq = np.sqrt(label_freq * (1 - label_freq) / 100) * self.z

            if (accuracy - limit_conf_accu) > (label_freq + limit_conf_freq):
                accepted_idx.append(index)

        return accepted_idx

    def is_rejected_y(self, labels_count, incorrect_labels, accuracy, num_processed, index):

        label_freq = labels_count[incorrect_labels[index]] / num_processed
        limit_conf_accu = np.sqrt(accuracy * (1 - accuracy) / 100) * self.z
        limit_conf_freq = np.sqrt(label_freq * (1 - label_freq) / 100) * self.z

        if (accuracy + limit_conf_accu) < (label_freq - limit_conf_freq):
            return True

        return False

    def fit(self, trn_data, labels):
        # First Iteration
        trn_data_keep = trn_data[0, :].reshape(1, len(trn_data[0, :]))
        labels_keep = np.array(labels[0]).reshape(1)
        classi_list = [[1]]

        labels_count = {}
        for label in np.unique(labels):
            labels_count[label] = 0

        for j in range(1, trn_data.shape[0]):

            # Step 1: Obtain the similarities with each sample at the Content Description
            similarities = [-self.d(trn_data[j, :], trn_sample) for trn_sample in trn_data_keep]

            labels_count[labels[j]] += 1

            # Step 2: Check that there are acceptable samples at the Content Description based on similarity scores
            accepted_ys = self.get_accepted_ys(labels_count, labels_keep,
                                               [np.mean(sample) for sample in classi_list], j)

            if len(accepted_ys) > 0:
                if len(accepted_ys) == 1:
                    neighbor = accepted_ys[0]
                else:
                    neighbor = accepted_ys[np.argmax([similarities[y] for y in accepted_ys])]
            else:
                # Step 2.1: randomly select a number in the range between 1 and the length of the Content Description
                ith = np.random.random_integers(1, len(trn_data_keep))

                # Step 2.2: Select the ith most similar value to X in the content description
                neighbor = np.argsort(similarities)[-ith]

            # Step 3: If classification is incorrect, add it to the Content Description
            if labels[j] == labels_keep[neighbor]:
                pass
            else:
                trn_data_concat = trn_data[j, :].reshape(1, len(trn_data[j, :]))
                trn_data_keep = np.concatenate((trn_data_keep, trn_data_concat), axis=0)
                labels_keep = np.concatenate((labels_keep, np.array(labels[j]).reshape(1)))
                classi_list.append([])

            # Step 4: For each sample in the Content Description:
            remove_idx = []
            for m in range(len(similarities)):
                if similarities[m] >= similarities[neighbor]:
                    # Step 4.1: Update y's classification record
                    if labels[j] == labels_keep[m]:
                        classi_list[m].append(1)
                    else:
                        classi_list[m].append(0)

                    # Step 4.2: If record is poor
                    if self.is_rejected_y(labels_count, labels_keep, np.mean(classi_list[m]), j, m):
                        remove_idx.append(m)
                        pass

            for m in sorted(remove_idx, reverse=True):
                classi_list.pop(m)
            trn_data_keep = np.delete(trn_data_keep, remove_idx, axis=0)
            labels_keep = np.delete(labels_keep, remove_idx, axis=0)

        self.trn_data = trn_data_keep
        self.trn_labels = labels_keep
        print(str(len(trn_data_keep)))



    def classify(self, tst_data):
        self.tst_labels = np.zeros((tst_data.shape[0], 1))
        for i in range(tst_data.shape[0]):
            a = tst_data[i,:]
            neighbor_idxs = np.argpartition([self.d(tst_data[i,:], trn_sample) for trn_sample in self.trn_data],
                                            kth=self.k-1)[:self.k]
            labels, counts = np.unique(self.trn_labels[neighbor_idxs], return_counts=True)
            self.tst_labels[i] = labels[np.argmax(counts)]