#####################################################################################################################################################

# preprocessing code for task 1
# this code is inspired from
# https://appliedmachinelearning.blog/2019/03/04/state-of-the-art-text-classification-using-bert-model-predict-the-happiness-hackerearth-challenge/

#####################################################################################################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import argparse
import os


def bertIn(path_):
    le = LabelEncoder()
    # read source data from csv file
    df_data = pd.read_csv(path_, sep='; ', engine='python')

    # split into train, test
    df_train, df_test = train_test_split(df_data, test_size=0.3, random_state=0, stratify=df_data.Gold.values)

    # create a new dataframe for train, dev data
    df_bert = pd.DataFrame({'id': np.arange(len(df_train)),
                            'label': le.fit_transform(df_train['Gold']),
                            'alpha': ['a'] * df_train.shape[0],
                            'text': df_train['Text']})

    # split into train, dev
    df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.3, random_state=0)

    # create new dataframe for test data
    df_bert_test = pd.DataFrame({'id': np.arange(len(df_test)),
                                 'text': df_test['Text'],
                                 'label': df_test['Gold']})

    # output tsv file, no header for train and dev
    if not os.path.exists('baseline/task1/data'):
        os.makedirs('baseline/task1/data')

    df_bert_train.to_csv('baseline/task1/data/train.tsv', sep='\t', index=False, header=False)
    print("Train data loaded")
    df_bert_dev.to_csv('baseline/task1/data/dev.tsv', sep='\t', index=False, header=False)
    print("Dev data loaded")
    df_bert_test.to_csv('baseline/task1/data/test.tsv', sep='\t', index=False, header=False)
    print("Test data loaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, default='./data/trial/fnp2020-fincausal-task1.csv',
                        help='task 1 data input repo')

    args = parser.parse_args()
    path_ = args.infile
    bertIn(path_)