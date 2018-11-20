import os
import re
import time

import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt

from eval_plot.evaluation import evaluate
from eval_plot.evaluation import ploting_v
from eval_plot.evaluation import ploting_v3d
from algorithms.kmeans import Kmeans
from algorithms.methods import compute_covariance
from algorithms.methods import proportion_of_variance
from preproc.preprocess import Preprocess
from sklearn.preprocessing.label import LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA



# ------------------------------------------------------------------------------------------------------- Read databases
def obtain_arffs(path):
    # Read all the databases
    arffs_dic = {}
    for filename in os.listdir(path):
        if re.match('(.*).arff', filename):
            arffs_dic[filename.replace('.arff', '')] = arff.loadarff(path + filename)
    return arffs_dic

# -------------------------------------------------------------------------------------------------------------- K-means
def tester_kmeans(data_x, groundtruth_labels):
    # HYPERPARAMETERS
    num_clusters = 2        # Number of clusters
    num_tries_init = 1      # Number of different initializations of the centroids
    max_iterations = 6     # Number of iterations for each initialization

    print('\033[1m' + 'Chosen HYPERPARAMETERS: ' + '\033[0m'+'\nNumber of clusters: '+str(
        num_clusters)+'\nNumber of different initilizations: '+str(num_tries_init)+'\nMaximum number of iterations '
                                                                            'per initialization: '+str(max_iterations))

    start_time = time.time()
    tst2 = Kmeans(num_clusters, num_tries_init, max_iterations)
    tst2.kmeans_method(data_x)
    print('Running time: %s seconds' % round(time.time() - start_time, 4))
    evaluate(tst2.labels_km, groundtruth_labels, data_x)
    return tst2.labels_km

# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    print('\033[1m' + 'Loading all the datasets...' + '\033[0m')
    arffs_dic = obtain_arffs('./datasets/')

    # Extract an specific database
    dataset_name = 'hypothyroid'       # possible datasets ('hypothyroid', 'breast-w', 'waveform')
    dat1 = arffs_dic[dataset_name]
    df1 = pd.DataFrame(dat1[0])     # original data in pandas dataframe
    groundtruth_labels = df1[df1.columns[len(df1.columns)-1]].values  # original labels in a numpy array
    df1 = df1.drop(df1.columns[len(df1.columns)-1],1)
    if dataset_name == 'hypothyroid':
        df1 = df1.drop('TBG', 1)    # This column only contains NaNs so does not add any value to the clustering
    data1 = df1.values              # original data in a numpy array without labels
    load = Preprocess()
    data_x = load.preprocess_method(data1)
    data_x = data_x.astype(np.float64)
    le = LabelEncoder()
    le.fit(np.unique(groundtruth_labels))
    groundtruth_labels = le.transform(groundtruth_labels)

    num_clusters = len(np.unique(groundtruth_labels)) # Number of different labels



# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()
