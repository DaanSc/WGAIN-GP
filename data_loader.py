# -*- coding: utf-8 -*-
"""
Created on het jaar 0

@author: Jezus?
"""

# Necessary packages
import numpy as np
import pandas as pd 

def data_split(splits,N, seed=123):
    obs=np.arange(0,N)
    np.random.shuffle(obs)
    batch=[]
    batch_size=(int)(np.ceil(N/splits))
    start=0
    end=batch_size
    for i in range(splits):
        new=obs[start:end]
        batch.append(new)
        start+=batch_size
        end+= batch_size
    return(batch)
    

def train_test_split(splits, N, seed):
    
    assert N%splits ==0,  "We cannot evenly divide by the number of splits"
    
    splitted_data=np.array(data_split(splits,N,seed=seed))
    all_groups= np.arange(0,splits)
    train=[]
    test=[]
    for split in all_groups:
        test.append(splitted_data[split])
        new_train=splitted_data[all_groups!=split]
        dim = (int) (np.shape(new_train)[0]*np.shape(new_train)[1])
        train.append(np.reshape(new_train,(dim)))
    return([train, test])

def ToyData(data_name):
  # Load data
  if data_name in ['letter', 'spam']: 
    file_name = 'Toydata/'+data_name+'.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    print(np.shape(data_x))
  return(data_x)

def BolData(data_name):
  # Load data
  if data_name in ['GAIN1', 'GAIN2', 'GAIN3',
                   'GAIN_return1', 'GAIN_return2', 'GAIN_return3',
                   'GAIN_case1', 'GAIN_case2', 'GAIN_case3']: 
    file_name = 'Data/'+data_name+'.csv'
    names_data= pd.read_csv('Data/'+data_name+'.csv',low_memory=False)
    names = names_data.head()
    names = list(names.columns)
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  return([data_x, names])

# Function which opens data
# def openData():
    # return(empty soul)

# Function which imputes data
def data_Imputer(data_x, miss_rate, cols, seed=123):
      
  # Parameters
  no, dim = data_x.shape
  
  np.random.seed(seed)
  
  # Introduce missing data
  data_m = np.ones((no,dim))
  miss_data_x = data_x.copy()
  obs = np.arange(0,no)
  np.random.shuffle(obs)
  obs=obs[0:(int)(round(miss_rate*no,0))]
  for col in cols:
      data_m[obs,col] = 0
      miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m  



























