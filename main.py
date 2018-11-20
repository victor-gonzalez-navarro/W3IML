import os
import re
import time

import pandas as pd
import numpy as np
from scipy.io import arff

from preproc.preprocess import Preprocess
from sklearn.preprocessing.label import LabelEncoder


# ------------------------------------------------------------------------------------------------------- Read databases
def obtain_arffs(path):
    # Read all the databases
    arffs_dic = {}
    for filename in os.listdir(path):
        if re.match('(.*).arff', filename):
            arffs_dic[filename.replace('.arff', '')] = arff.loadarff(path + filename)
    return arffs_dic

# ------------------------------------------------------------------------------------------------------------Algorithms


# ----------------------------------------------------------------------------------------------------------------- Main
def main():
    print('\033[1m' + 'Loading all the datasets...' + '\033[0m')
    arffs_dic = obtain_arffs('./datasets/')

    # Extract an specific database
    dataset_name = 'hypothyroid'
    dat1 = arffs_dic[dataset_name]
    df1 = pd.DataFrame(dat1[0])     # original data in pandas dataframe
    groundtruth_labels = df1[df1.columns[len(df1.columns)-1]].values  # original labels in a numpy array
    df1 = df1.drop(df1.columns[len(df1.columns)-1],1)

    data1 = df1.values              # original data in a numpy array without labels
    load = Preprocess()
    data_x = load.preprocess_method(data1)
    #data_x = data_x.astype(np.float64)

    le = LabelEncoder()
    le.fit(np.unique(groundtruth_labels))
    groundtruth_labels = le.transform(groundtruth_labels)
    num_clusters = len(np.unique(groundtruth_labels)) # Number of different labels



# ----------------------------------------------------------------------------------------------------------------- Init
if __name__ == '__main__':
    main()
