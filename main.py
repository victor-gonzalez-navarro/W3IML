import os
import re
import time

import pandas as pd
import numpy as np
from scipy.io import arff

from preproc.preprocess import Preprocess
from algorithms.ib1Algorithm import ib1Algorithm
from algorithms.ib2Algorithm import ib2Algorithm
from algorithms.ib3Algorithm import ib3Algorithm
from sklearn.preprocessing.label import LabelEncoder


# -------------------------------------------------------------------------------------------------------- Read datasets
def obtain_arffs(path):
    # Read all the datasets
    processed = []
    arffs_dic = {}
    folds_dic = {}

    for folder in os.listdir(path):
        for filename in os.listdir(path + folder + '/'):
            if re.match('(.*).fold.(\d*).(train|test).arff', filename) and filename not in processed:
                row = int(re.sub('(\w*).(\d*).(\w*)', r'\2', filename))
                trn_file = re.sub('(train|test)', 'train', filename)
                tst_file = re.sub('(train|test)', 'test', filename)
                folds_dic[row] = []
                folds_dic[row].append(arff.loadarff(path + folder + '/' + trn_file)[0])
                folds_dic[row].append(arff.loadarff(path + folder + '/' + tst_file)[0])
                processed.append(trn_file)
                processed.append(tst_file)
        arffs_dic[folder] = folds_dic
    return arffs_dic

def trn_tst_idxs(ref_data, dataset):
    trn_tst_dic = {}
    for key, fold_data in dataset.items():
        trn_idxs = [np.where(ref_data == sample)[0][0] for sample in fold_data[0]]
        tst_idxs = [np.where(ref_data == sample)[0][0] for sample in fold_data[1]]
        trn_tst_dic[key] = []
        trn_tst_dic[key].append(trn_idxs)
        trn_tst_dic[key].append(tst_idxs)
    return trn_tst_dic

# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    print('\033[1m' + 'Loading all the datasets...' + '\033[0m')
    arffs_dic = obtain_arffs('./datasetsSelected/')

    # Extract an specific database
    dataset_name = 'grid'
    dataset = arffs_dic[dataset_name]

    ref_data = np.concatenate((dataset[0][0], dataset[0][1]), axis=0)
    trn_tst_dic = trn_tst_idxs(ref_data, dataset)

    df1 = pd.DataFrame(ref_data)
    groundtruth_labels = df1[df1.columns[len(df1.columns) - 1]].values  # original labels in a numpy array
    df1 = df1.drop(df1.columns[len(df1.columns) - 1], 1)

    # ------------------------------------------------------------------------------------------------------- Preprocess
    data1 = df1.values  # original data in a numpy array without labels
    load = Preprocess()
    data_x = load.preprocess_method(data1)

    # ---------------------------------------------------------------------------------------- Encode groundtruth labels
    le = LabelEncoder()
    le.fit(np.unique(groundtruth_labels))
    groundtruth_labels = le.transform(groundtruth_labels)

    # -------------------------------------------------------------------------------------------- Supervised classifier
    accuracies = []
    for trn_idxs, tst_idxs in trn_tst_dic.values():
        trn_data = data_x[trn_idxs]
        trn_labels = groundtruth_labels[trn_idxs]
        tst_data = data_x[tst_idxs]
        tst_labels = groundtruth_labels[tst_idxs]

        knn = ib3Algorithm(k=1, metric='euclidean')
        knn.fit(trn_data, trn_labels)
        knn.classify(tst_data)

        accuracies.append((sum([a == b for a, b in zip(tst_labels, knn.tst_labels)]))/len(tst_labels))

    print('The accuracy of classification is: ' + str(round(np.mean(accuracies),3)) + ' ± ' + str(round(np.std(
        accuracies),2)))
    print('The algorithm has finished successfully')

# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()
