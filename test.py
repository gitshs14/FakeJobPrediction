import pandas as pd
import numpy as np
'''
dataset = pd.read_csv("Dataset/DataSet.csv")
Y = dataset['fraudulent']

fraud = dataset.loc[dataset['fraudulent'] == 'f']
true_job = dataset.loc[dataset['fraudulent'] == 't']

fraud = fraud[:1000]
print(fraud.shape)
dataset = pd.concat([fraud, true_job], ignore_index=True)
dataset.to_csv("temp.csv",index=False)
'''

dataset = pd.read_csv("temp.csv")
Y = dataset['fraudulent']
print(dataset.shape)

unique,count = np.unique(Y, return_counts=True)
print(unique)
print(count)
print(dataset)
