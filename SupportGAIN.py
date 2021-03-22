# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 9:11:01 2021

@author: De Stille Fotograaf
"""
# External packages
import numpy as np
import tensorflow as tf
# imports from files that are contained in map
from utils import renormalization, rounding, binary_sampler

# Additional functions
# Function that creates batches
def CreateBatches(N, batch_size):
    obs = np.arange(0,N)
    np.random.shuffle(obs)
    No_batches=(int)(np.ceil(len(obs)/batch_size))
    batches=[]
    start=0
    end= batch_size
    for i in range(No_batches):
        new= obs[start:end]
        start += batch_size
        end += batch_size
        batches.append(new)
    return(batches)
    
# Hint matrix, we provide two options:
    # 1. Random hint matrix
    # 2. Conditioned hint matrix
def Hint_Matrix(Hint_Rate,M,cols, conditioned=True):
    Obs, Dim= np.shape(M)
    Hint = M.copy()
    if (conditioned == True):
        for col in cols:
            B = np.reshape(np.array(binary_sampler(Hint_Rate, Obs,1)),(Obs))
            Hint[:,col]= B*M[:,col] + 0.5*(1-B)
    if(conditioned==False):
        B= np.reshape(np.array(binary_sampler(Hint_Rate, Obs,Dim)),(Obs, Dim))
        Hint= B*M + 0.5*(1-B)
    return(Hint)

def MissingColumns(M):
    Obs, Dim= np.shape(M)
    cols = []
    for i in range(Dim):
        if(np.sum(M[:,i])!= Obs):
            cols.append(i)
    cols=np.array(cols)
    return(cols)

# Creates lists for accuracy plots
def AccuracyNames(margins, Impute_Cols):
    names=[]
    mc=0
    for i in range(len(Impute_Cols)):
        for j in range(len(margins)):
            Fun_Name = 'C%i' %Impute_Cols[i] + 'M%i' %margins[j] 
            names.append(Fun_Name)
            names[mc] = [names[mc]]
            mc+=1
    return(names)
   
# Define correct accuracy of imputations based on a certain margin
# margin=0 exactly correct, margin=1, 1 deviation in absolute value, etc. 
def Correct_Imputations(compare,Impute_Cols, margins):
  all_correct= []
  for margin in margins:
      correct=[]
      for col in range(len(Impute_Cols)):
          good= np.sum(np.abs(compare[col][0,:]-compare[col][1,:])<=margin)/np.shape(compare[col])[1]
          correct.append(good)
      correct=np.array(correct)
      all_correct.append(correct)
    
  n_correct= all_correct[0]
  for j in range(1,len(margins)):
    n_correct=np.vstack((n_correct,all_correct[j]))
      
  return(n_correct)    
   
# Functions adds accuracies for new iteration      
def AddAccuracy(names, n_correct, cols, margins):
    mc=0        
    for i in range(len(cols)):
        for j in range(len(margins)):
            names[mc].append(n_correct[j,i])
            mc+=1
    return(names)



# Impute function
def Impute(data_x, X, M, generator, Norm_Parameters):
    Gen_Input = np.hstack((X,M ))
    Gen_Input=  tf.convert_to_tensor(Gen_Input)
    Imputed_Data= generator(Gen_Input, training=False)
    Imputed_Data = M * X + (1-M) * Imputed_Data  
    Imputed_Data= np.array(Imputed_Data)
    # Renormalization
    Imputed_Data = renormalization(Imputed_Data, Norm_Parameters)      
    # Rounding
    Imputed_Data = rounding(Imputed_Data, data_x) 
    return(Imputed_Data)

# Function which returns the actual missing values and the imputed values
def Compare(Act_X,X,M, cols, generator, Norm_Parameters):
    Col_Dim=len(cols)
    Actual=Act_X[M==0]
    Actual= np.reshape(Actual, ((int)(np.shape(Actual)[0]/Col_Dim),Col_Dim))
    Imputed = Impute(Act_X,X,M, generator, Norm_Parameters)[M==0]
    Imputed= np.reshape(Imputed, ((int)(np.shape(Imputed)[0]/Col_Dim),Col_Dim))
    compare= []
    RMSE=0
    for col in range(Col_Dim):
        new=np.vstack((Actual[:,col],Imputed[:,col]))
        RMSE += np.sum((Imputed[:,col]-Actual[:,col])**2)
        compare.append(new)
    T= np.shape(Imputed)[0]*np.shape(Imputed)[1]
    RMSE = (RMSE/T)**(1/2)
    return([compare,RMSE])

# Function that is used to compute the accuracies on the test set
def accuracy_test(compare_test, Measurement,beta=1):
    # Compate accuracy measures
    ind_accs=np.zeros(len(compare_test))
    i=0
    count=0
    # Compute accuracy measure for every imputed column
    for dist in compare_test:
        if(Measurement[i]=="RMSE"): 
            ind_accs[i]= round((np.sum((dist[0]-dist[1])**2)/len(dist[0]))**(1/2),3)
            count+=1
        # Compute F-score
        if(Measurement[i]!="RMSE"):
            # Compute F-scores for label 0 and label 1 and take then the average
            actual= dist[0]
            imputed= dist[1]
            precision1_elements = np.where(imputed==1) 
            precision1= np.sum(actual[precision1_elements]== imputed[precision1_elements])/len(np.array(precision1_elements)[0])
            recall1_elements = np.where(actual==1)
            recall1= np.sum(actual[recall1_elements]== imputed[recall1_elements])/len(np.array(recall1_elements)[0])
            F1 = np.array((1+beta**2)* (recall1*precision1)/(beta**2*precision1 + recall1))
            precision0_elements = np.where(imputed==0) 
            precision0= np.sum(actual[precision0_elements]== imputed[precision0_elements])/len(np.array(precision0_elements)[0])
            recall0_elements = np.where(actual==0)
            recall0= np.sum(actual[recall0_elements]== imputed[recall0_elements])/len(np.array(recall0_elements)[0])
            F0 = np.array((1+beta**2)* (recall0*precision0)/(beta**2*precision0 + recall0))
            if(np.isnan(F1)==True):
                F1=0
            if(np.isnan(F0)==True):
                F0=0
            ind_accs[i]= (F1+F0)/2
        i+=1
    return(ind_accs)

