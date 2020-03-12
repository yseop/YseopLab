#!/usr/bin/env python
# coding: utf-8

####################################################

## post-processing script that takes in BERT results
## and generates readable predictions file

####################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame

def bertPrepOutput(): 

	df_bert_test = pd.read_csv('./data/test.tsv', sep='\t')
	
	# convert output to csv format

	df_results = pd.read_csv("./bert/bert_output/test_results.tsv",sep="\t",header=None)
	df_results_csv = pd.DataFrame({'id':df_bert_test['id'],
                               'label':df_results.idxmax(axis=1)})
							   
	# count number of unique labels to determine number of classes
	df_results_csv['label'].nunique()
 

	# writing into .csv
	df_results_csv.to_csv('./data/result.csv',sep=",",index=None)
	print("Results loaded")


if __name__ == "__main__":
	bertPrepOutput()