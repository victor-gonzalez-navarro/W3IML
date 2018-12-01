import os
import re
import time
import math

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

        foldata_trn = pd.DataFrame(fold_data[0])
        foldata_trn = foldata_trn.fillna('nonna')
        foldata_trn = foldata_trn.values

        foldata_tst = pd.DataFrame(fold_data[1])
        foldata_tst = foldata_tst.fillna('nonna')
        foldata_tst = foldata_tst.values

        trn_idxs = [np.where((ref_data == sample).all(axis=1))[0][0] for sample in foldata_trn]
        tst_idxs = [np.where((ref_data == sample).all(axis=1))[0][0] for sample in foldata_tst]
        trn_tst_dic[key] = []
        trn_tst_dic[key].append(trn_idxs)
        trn_tst_dic[key].append(tst_idxs)
    return trn_tst_dic

# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    print('\033[1m' + 'Loading all the datasets...' +'\033[0m')
    arffs_dic = obtain_arffs('./datasetsSelected/')

    # Extract an specific database
    dataset_name = 'hepatitis'
    dataset = arffs_dic[dataset_name]

    # Use folder 0 of that particular dataset to find indices of train and test
    ref_data = np.concatenate((dataset[0][0], dataset[0][1]), axis=0)

    # --------------------------------------------------------------------------------- To compute indices for each fold
    df_aux = pd.DataFrame(ref_data)
    df_aux = df_aux.fillna('nonna').values
    trn_tst_dic = trn_tst_idxs(df_aux, dataset)

    # ------------------------------------------------------------------------------------------------------- Preprocess
    df1 = pd.DataFrame(ref_data)
    groundtruth_labels = df1[df1.columns[len(df1.columns) - 1]].values  # original labels in a numpy array
    df1 = df1.drop(df1.columns[len(df1.columns) - 1], 1)

    data1 = df1.values  # original data in a numpy array without labels
    load = Preprocess()
    data_x = load.preprocess_method(data1)

    # ---------------------------------------------------------------------------------------- Encode groundtruth labels
    le = LabelEncoder()
    le.fit(np.unique(groundtruth_labels))
    groundtruth_labels = le.transform(groundtruth_labels)

    accuracies = []
    fold_number = 0

    # ---------------------------------------------------------------- Reading K value and distance metric from keyboard
    print('\n'+'\033[1m'+'Which K value do you want to use?'+'\033[0m')
    k = int(input('Insert a number between 1-10: '))
    print('\n'+'\033[1m'+'Which distance function do you want to use?'+'\033[0m'+'\n1: Euclidean\n2: Manhattan')
    dist = int(input('Insert a number between 1-2: '))
    if dist == 1:
        metric = 'euclidean'
    elif dist == 2:
        metric = 'manhattan'
    print('\n'+'\033[1m'+'Which voting policy do you want to use?'+'\033[0m'+'\n1: Most voted solution\n2: Modified '
                                                                             'Plurality\n3: Borda Count')
    voting_policy = int(input('Insert a number between 1-3: '))
    print('')
    if voting_policy == 1:
        voting_policy = 'most_voted'
    elif voting_policy == 2:
        voting_policy = 'modified_plurality'
    elif voting_policy == 3:
        voting_policy = 'borda_count'

    # -------------------------------------------------------------------------------------------- Supervised classifier
    for trn_idxs, tst_idxs in trn_tst_dic.values():
        fold_number = fold_number +1
        print('Computing accuracy for fold number '+str(fold_number))
        trn_data = data_x[trn_idxs]
        trn_labels = groundtruth_labels[trn_idxs]
        tst_data = data_x[tst_idxs]
        tst_labels = groundtruth_labels[tst_idxs]

        knn = ib2Algorithm(k, metric, voting_policy)
        knn.fit(trn_data, trn_labels)
        knn.classify(tst_data)

        accuracies.append((sum([a == b for a, b in zip(tst_labels, knn.tst_labels)]))/len(tst_labels))

    mean_accuracies = str(round(np.mean(accuracies),3))
    std_accuracies = str(round(np.std(accuracies),2))
    print('\033[1m'+'The mean accuracy of classification in the test set is: ' + mean_accuracies + ' ± ' + std_accuracies+'\033[0m')

# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()
