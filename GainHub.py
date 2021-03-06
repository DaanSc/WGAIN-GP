# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 9:11:01 2021

@author: Het Gespleten Geweten 
"""

# RUN THIS BLOCK FIRST!!!! --> It loads all the functions and standard settings
# Standard settings: 
    # 1. data --> data for 2 gains
    # 2. K --> Number of folds
    # 3. missrates --> missingrates for training and test
    # 4. ceiling --> ceilings for embeddings
    # 5. plots --> plot options

# Additionally, we have the following blocks:
    # 1: Settings for GAIN,WGAIN, WGAIN-GP for imputation of Shipment Days and Transporter Type
    # 2: GAIN imputation for Shipment Days and Transporter Type
    # 3: WGAIN imputation for Shipment Days and Transporter Type
    # 4: WGAIN-GP imputation for Shipment Days and Transporter Type
    # 5: Settings for GAIN,WGAIN, WGAIN-GP for imputation of Transporter Days
    # 6: GAIN imputation for imputation of Transporter Days
    # 7: WGAIN imputation for imputaion of Transporter Days
    # 8: WGAIN-GP imputation for imputation of Transporter Days

# Necessary packages
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

# imports from files that are contained in map
from data_loader import  BolData
from veinGAIN import CV

# This file will be used to obtain GAINS which will be used for the main part of the model
# The code for the main part is written in another file. In this file, we apply validation methods for different 
# configurations of the GAINS. We use RMSE/accuracy as measures to compare the GAINS with. 
# The procedure is as follows: 
    # 1. We import a data-set from the file data_loader. Additionally, we create K-folds such that we are able to 
    #    distinguish between test and training data. 
    # 2. We set the parameters used for training. These include:
        # 1. CV_Complete: boolean which indicates whether we should apply test-training splits on all folds, or only the first
        # 2. Missing-rates: Imputation probabilities for training and test. By default for training, we select 0.5 and for test 0.8
        # 3. Ceiling: Maximum output layer for embeddings --> The dimension reduction of our categorical variables
        # 4. Plot_booleans: Indicate wheter the results on the test set and training set should be plotted. This can be set per fold
        # 5. Imputation columns: The user must enter the column numbers to impute the data from. Addtionally, the names of the columns
        #    and accuracy measurement for the test test must be stated. 
        # 6. Embeddings: If embedding is used, the columns numbers of these columns should be given. 
        # 7. GAIN-type: Select the type of GAIN. User can decide between GAIN and WGAIN-GP
        # 8. Hyperparamters: The hyperparameters that might change to improve the performance of the GAIN model
        # 9. Epochs: Number of training epochs for training one GAIN. 
# In case different columns have different accuracy measurements, we will report the best configurations for the two methods seperately,
# since the results are non-compareable. In the end the configurations with highest accuracy/lowest loss are returned as well as the generator 
# which was formed using this configuration. 

# IF we want TO prevent warnings when saving generators (which stem from the fact that pandas is enabled)
# run this file before running CrossRoads!


# Set here your output path for images, CV progress, embeddings and generators. End with /. 
# Example: "C:/Users/map/"
path= "C:/Users/Lars Hurkmans/Downloads/GAIN_CODE"
out_path= path +"/Plots/"
CV_path = path + "/CV/"
generator_path= path+ "/Generators/"

assert out_path != None, "Assign output paths"

# Data sets
all_data1, names1= BolData("GAIN1")
return1, names2= BolData("GAIN_return1")
case1, names3= BolData("GAIN_case1")
return2, names4= BolData("GAIN_return2")
case2, names5= BolData("GAIN_case2")

# K-fold CV
K=5

# Parameters
# Missing rates for training and test
train_missrate=0.5
test_missrate=0.9

# Ceiling for embeddings --> The maximum number of output layers. This can be set
# per variable
ceiling=[1,1,1,1]

# Plot test results
Test_Plot=[True, True, True, True, True]
# State whether Progression of training GAIN needs to be plotted.
# Note that this can be set per fold. 
Plot_Gain=[True, True, True, True, True]


#%%
# BLOCK 1: 
# Embedding Dataset 1 --> Settings for GAIN and WGAIN
# Determine columns of which we want to impute
Impute_Cols=[5]
# Variable names of imputed colummns
Var_Names=np.array(names1)[Impute_Cols]
# accuracy measuement for test set: Select RMSE or ACC
Measurement= ["ACC"]

# Determine columns to Embed, in case wanted
Embed_Cols=[6,15,16,17]
Embed_Names=np.array(names1)[Embed_Cols]

# List of hyperparameters
# Batch_size: Number of observations that are trained at once
# Hint_rate: The proportion of hints given to the discriminator. 
#            The higher this value, the les less 0.5 values are given
# Lamba's is a list of hyperparameters for Generator and Discriminator Loss, 
# where: 
    # 1. Lambda[0]=  hyperparameter for MSE, which is used as part of generator loss
    # 2. Lambda[3] = hyperparameter for gradient penalty (only for WGAIN-GP)
# In case we are not considering, WGAIN-GP, only first element will be taken
# Seed: Although not really a hyperparameter, the seed might influence the training process.
#       Hence, we will try out several seeds to visually detect mode collapse. 
# We can try out different configurations, which is why several values are provided as input
hyperparameters= {'batch_size': [2048],
                          'hint_rate': [0.5],
                          'Lambda[0]': [1000],
                          'Lambda[3]': [10,],
                          "seed": [5]}
# Set number of training Epochs
training_epochs=300
#%%
# BLOCK 2:
# Obtain results for Shipment Days and Transporter Type using GAIN
print("Begin GAIN Training on first dataset")
GAIN_Train1= CV(all_data1, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="GAIN",
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_shipment = GAIN_Train1[0][0]
generator_shipment.save(generator_path+"GAIN/Del")
#%%
# BLOCK 3:
# Obtain results for Shipment Days and Transporter Type using WGAIN
print("Begin WGAIN Training on first dataset")
WGAIN_Train1= CV(all_data1, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAIN", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
# NOTE: Too long output path (>16 characters after generator_path) gives errors. Why?
generator_shipment = WGAIN_Train1[0][0]
generator_shipment.save(generator_path+"WGAIN/Del")
#%%
# BLOCK 4:
# For WGAIN-GP, we add additional hyperparameters in the form of lambda_3 (lamba_gp)
print("Start WGAIN-GP training on first dataset")
WGAINGP_Train1= CV(all_data1, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAINGP", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
# NOTE: Too long output path (>16 characters after generator_path) gives errors. Why?
generator_shipment = WGAINGP_Train1[0][0]
generator_shipment.save(generator_path+"WGAINGP/Del")
#%% 
# Block 5: Return and case 2
# settings for all situations
# Determine columns of which we want to impute
Impute_Cols=[14]    

# accuracy measuement for test set: Select RMSE or ACC
Measurement= ["ACC"]

# Determine columns to Embed, in case wanted
Embed_Cols=[5,15,16,17]
Embed_Names=np.array(names4)[Embed_Cols]

hyperparameters= {'batch_size': [2048],
                          'hint_rate': [0.5],
                          'Lambda[0]': [100],
                          'Lambda[3]': [10,],
                          "seed": [5]}

# Set number of training Epochs
training_epochs= 10
#%%
# Block 6: Return 2 WGAINGP
Var_Names= ["Return 2"]
print("Start WGAINGP training on return 2")
WGAINGP_return2= CV(return2, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAINGP", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_return2 = WGAINGP_return2[0][0]
generator_return2.save(generator_path+"WGAINGP/Return2")
# Block 7: case 2 WGAINGP
Var_Names= ["Case 2"]
print("Start WGAINGP training on case 2")
WGAINGP_case2= CV(case2, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAINGP", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_case2 = WGAINGP_case2[0][0]
generator_case2.save(generator_path+"WGAINGP/Case2")
#%%
# Block 8: Return 2 WGAIN
Var_Names= ["Return 2"]
print("Start WGAIN training on return 2")
WGAIN_return2= CV(return2, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAIN", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_return2 = WGAIN_return2[0][0]
generator_return2.save(generator_path+"WGAIN/Return2")
# Block 9: case 2 WGAIN
Var_Names= ["Case 2"]
print("Start WGAIN training on case 2")
WGAIN_case2= CV(case2, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAIN", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_case2 = WGAIN_case2[0][0]
generator_case2.save(generator_path+"WGAIN/Case2")
#%%
# Block 10: Return 2 GAIN
Var_Names=["Return 2"]
print("Start GAIN training on return 2")
GAIN_return2= CV(return2, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="GAIN", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_return2 = GAIN_return2[0][0]
generator_return2.save(generator_path+"GAIN/Return2")
# Block 11: case 2 GAIN
Var_Names=["Case 2"]
print("Start GAIN training on case 2")
GAIN_case2= CV(case2, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="GAIN", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_case2 = GAIN_case2[0][0]
generator_case2.save(generator_path+"GAIN/Case2")
#%%

# Determine columns of which we want to impute
Impute_Cols=[14]    

# accuracy measuement for test set: Select RMSE or ACC
Measurement= ["ACC"]

# Determine columns to Embed, in case wanted
Embed_Cols=[5,15,16]
Embed_Names=np.array(names4)[Embed_Cols]

hyperparameters= {'batch_size': [2048],
                          'hint_rate': [0.5],
                          'Lambda[0]': [100],
                          'Lambda[3]': [10,],
                          "seed": [5]}
# Set number of training Epochs
training_epochs= 300
#%%

# Block 12: Return 1 WGAINGP
Var_Names= ["Return 1"]
print("Start WGAINGP training on return 1")
WGAINGP_return2= CV(return1, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAINGP", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_return2 = WGAINGP_return2[0][0]
generator_return2.save(generator_path+"WGAINGP/Return1")
# Block 13: case 1 WGAINGP
Var_Names= ["Case 1"]
print("Start WGAINGP training on case 1")
WGAINGP_case2= CV(case1, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAINGP", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_case2 = WGAINGP_case2[0][0]
generator_case2.save(generator_path+"WGAINGP/Case1")
#%%

# Block 14: Return 1 WGAIN
Var_Names= ["Return 1"]
print("Start WGAIN training on return 1")
WGAIN_return2= CV(return1, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAIN", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_return2 = WGAIN_return2[0][0]
generator_return2.save(generator_path+"WGAIN/Return1")
# Block 15: case 1 WGAIN
Var_Names= ["Case 1"]
print("Start WGAIN training on case 1")
WGAIN_case2= CV(case1, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="WGAIN", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_case2 = WGAIN_case2[0][0]
generator_case2.save(generator_path+"WGAIN/Case1")
#%%
# Block 16: Return 1 GAIN
Var_Names=["Return 1"]
print("Start GAIN training on return 1")
GAIN_return2= CV(return1, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="GAIN", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_return2 = GAIN_return2[0][0]
generator_return2.save(generator_path+"GAIN/Return1")
# Block 17: case 1 GAIN
#%%
Var_Names=["Case 1"]
print("Start GAIN training on case 1")
GAIN_case2= CV(case1, K=K, CV_Complete=False, train_missrate=train_missrate, test_missrate=test_missrate,
    ceiling=ceiling, conditioned_critic=False, conditioned_MSE= False, Test_Plot=Test_Plot, 
    Plot_Gain=Plot_Gain, Impute_Cols=Impute_Cols, Var_Names=Var_Names, Measurement=Measurement, 
    Embed_Cols=Embed_Cols, Embed_Names=Embed_Names, hyperparameters=hyperparameters, 
    training_epochs=training_epochs, GAIN_type="GAIN", 
    out_path=out_path, generator_path=generator_path, CV_path=CV_path)
generator_case2 = GAIN_case2[0][0]
generator_case2.save(generator_path+"GAIN/Case1")





