# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:32:55 2021

@author: De stem van het Zuiden <:>
"""
# In this file, functions are stored which are used for SupportGAIN
import numpy as np
import pandas as pd
import time
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
# imports from files that are contained in map
from utils import normalization
from CompleteGAIN import GAIN
from CompleteWGAINGP import WGAINGP
from CompleteWGAIN import WGAIN
from DeStilleFotograaf import PlotHists, PRD_Plot, Progress_Plot
from EmbedPret import Embedded_GainInput
from EmbedPret import Embed_Compare
from data_loader import data_Imputer

# Function for gain
def gain(actual_data, impute_data, batch_size, hint_rate, alpha, ceiling, epochs, conditioned_hints=True, 
             conditioned_discriminator= False, conditioned_MSE= False, Embed_Cols=[], Var_Names=[], 
             Plot=False, out_path= None, generator_path=None, seed=123,):
        
    gain_parameters = {'batch_size': batch_size,
                          'hint_rate': hint_rate,
                          'alpha': alpha,
                          'epochs': epochs,
                          'conditioned_hints': conditioned_hints,
                          'conditioned_discriminator': conditioned_discriminator,
                          'conditioned_MSE': conditioned_MSE,}
    
    model = GAIN(actual_data, impute_data, gain_parameters, ceiling=ceiling,
                 Embed_Cols=Embed_Cols, Var_Names=Var_Names, Plot=Plot, out_path= out_path, 
                  generator_path=generator_path, seed=seed)

    return([gain_parameters,model])


# Function for wgain(-gp)
def wgaingp (actual_data, impute_data, batch_size, hint_rate, alpha, lambda_gp,  ceiling, epochs, conditioned_hints=True, 
             conditioned_critic= False, conditioned_MSE= False, Embed_Cols=[], Var_Names=[], 
             Plot=False, out_path= None, generator_path=None, seed=123,
             GAIN_type="WGAIN"):
  
  WGAINGP_parameters = {'batch_size': batch_size,
                    'hint_rate': hint_rate,
                    'alpha': alpha,
                    'lambda_gp': lambda_gp,
                    'conditioned_hints': conditioned_hints,
                    'conditioned_critic': conditioned_critic,
                    'conditioned_MSE': conditioned_MSE,
                    'epochs': epochs}
  if(GAIN_type=="WGAIN"):
      generator = WGAIN(actual_data, impute_data, WGAINGP_parameters, ceiling=ceiling, 
                        Embed_Cols=Embed_Cols, Var_Names=Var_Names,
                        Plot=Plot, out_path=out_path,generator_path=generator_path, seed=seed)
      
  if(GAIN_type=="WGAINGP"):
      generator = WGAINGP(actual_data, impute_data, WGAINGP_parameters, ceiling=ceiling, 
                        Embed_Cols=Embed_Cols, Var_Names=Var_Names,
                        Plot=Plot, out_path=out_path,generator_path=generator_path, seed=seed)
  return [WGAINGP_parameters, generator]


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

# Cross validation function
def CV(all_data, K, CV_Complete, train_missrate, test_missrate, ceiling, conditioned_critic, conditioned_MSE, 
       Test_Plot, Plot_Gain, Impute_Cols, Var_Names, Measurement, Embed_Cols, Embed_Names,
       hyperparameters, training_epochs, GAIN_type="GAIN", out_path=None, generator_path=None, CV_path=None):
    
    # Store results for all folds. These results are:
        # 1. The hyperparameters
        # 2. Error/Accuracy metric
        # 3. Number of effective epochs
        # 4. Running time
    # Number of columns, here we store the results
    Ncols = 2*len(Measurement) + 7
    # Number of rows=K
    Nrows= K
    
    # Initilaise configuration number
    Setting = 1
    
    # Track best accuracy measures
    besties=np.full(len(Impute_Cols),np.inf)
    M= np.array(Measurement)
    besties[M!="RMSE"] = -np.inf
    # Best generators
    best_generators = list(np.zeros(len(Impute_Cols)))
    # AUC of best generator
    best_AUC = np.zeros(len(Impute_Cols))
    # Hyperparameters of best configuration
    best_hypers= np.zeros((len(Impute_Cols),3))
    # Best configurations 
    best_configs = np.zeros(len(Impute_Cols))
    # Loop through all configurations
    for bs in range(len(hyperparameters["batch_size"])):
        for lambda_gp in range(len(hyperparameters["Lambda[3]"])):
            for lamba_MSE in range(len(hyperparameters["Lambda[0]"])):
                print("CV under configuration ", Setting)
        
                # Selected hyperparamters for gain                          
                batch_size=hyperparameters["batch_size"][bs]
                hint_rate=hyperparameters["hint_rate"][0]
                lambda_0=hyperparameters["Lambda[0]"][lamba_MSE]
                lambda_3=hyperparameters["Lambda[3]"][lambda_gp]
                seed=hyperparameters["seed"][0]
                            
                # Track best accuracy measures under current configuration
                besties_setting=np.full(len(Impute_Cols),np.inf)
                M= np.array(Measurement)
                besties_setting[M!="RMSE"] = -np.inf
                # Best generators
                best_generators_setting = list(np.zeros(len(Impute_Cols)))
                best_AUC_setting=np.zeros(len(Impute_Cols))
                # Store average results over the folds in current setting
                setting_acc = np.zeros(len(M))
                
                # CV-training for one configuration type of the hyperparamters
                if (CV_Complete==False):
                    np.random.seed(seed)
                    size=1/K
                    y=np.ones(np.shape(all_data)[0])
                    train_data, test_data, y_train, y_test=train_test_split(all_data,y, test_size=size)
                    train_actual, train_impute, train_mask=data_Imputer(train_data, train_missrate, Impute_Cols, seed=seed)
                    test_actual, test_impute, test_mask=data_Imputer(test_data, test_missrate, Impute_Cols, seed=seed)
                    # Transform test dataset
                    # GAIN training
                    
                    fold=0
                    
                    # Store results
                    Results = np.zeros((1,Ncols))
                    
                    # Store hyperparameters in results
                    Results[fold, 0] = batch_size
                    Results[fold, 1] = hint_rate 
                    Results[fold, 2] = lambda_0
                    Results[fold, 3] = lambda_3
                    Results[fold, 4] = seed
                    
                    # Train corresponding GAIN model. Either with embeddings or not. 
                    test_normalised, test_paras= normalization(test_impute)
                    
                    # Generator input 
                    Generator_Input, Embedded_Imputations, Layer_Size, Dim, Norm_Data_Rest, Norm_Pars, Data_Rest, M_Rest = Embedded_GainInput(test_impute, Embed_Cols, 
                                                                                                           Impute_Cols, ceiling, seed=seed )
                    # Select which GAIN method to use
                    if(GAIN_type=="GAIN"):
                        train_gain =gain(train_actual, train_impute, batch_size, hint_rate, lambda_0, epochs=training_epochs, ceiling=ceiling, 
                                       conditioned_hints=True, conditioned_discriminator=conditioned_critic, conditioned_MSE=conditioned_MSE, 
                                       Embed_Cols=Embed_Cols,Var_Names=Var_Names, Plot=Plot_Gain[fold], out_path=out_path, generator_path=generator_path)
                     
                    if(GAIN_type=="WGAIN"):
                        train_gain =wgaingp(train_actual, train_impute, batch_size, hint_rate, lambda_0, lambda_3, epochs=training_epochs, ceiling=ceiling, 
                                       conditioned_hints=True, conditioned_critic=conditioned_critic, conditioned_MSE=conditioned_MSE, 
                                       Embed_Cols=Embed_Cols,Var_Names=Var_Names, Plot=Plot_Gain[fold], out_path=out_path, generator_path=generator_path,
                                       GAIN_type=GAIN_type)
                    if(GAIN_type=="WGAINGP"):
                        train_gain= wgaingp(train_actual, train_impute, batch_size, hint_rate, lambda_0, lambda_3, epochs=training_epochs, ceiling=ceiling, 
                                       conditioned_hints=True, conditioned_critic=conditioned_critic, conditioned_MSE=conditioned_MSE, 
                                       Embed_Cols=Embed_Cols,Var_Names=Var_Names, Plot=Plot_Gain[fold], out_path=out_path, generator_path=generator_path,
                                       GAIN_type=GAIN_type)
                    
                    # Extract generators and obtain compare results
                    generators = []
                    compare_test = []
                    for variable in range(len(Impute_Cols)):
                        generator=train_gain[1][1][variable]
                        generators.append(generator)
                        compare_test.append(Embed_Compare(Generator_Input, Embedded_Imputations, Layer_Size, Dim,
                                         Norm_Data_Rest,  Norm_Pars, Data_Rest, M_Rest, test_actual, test_mask,Impute_Cols,generator))
                    effective_epochs = train_gain[1][2]
                    running_time =train_gain[1][3]
                    AUC=train_gain[1][4]
                                
                    # Compute accuracies
                    accuracies = np.zeros(len(Impute_Cols))
                    for variable in range(len(Impute_Cols)):
                        accuracies[variable]=accuracy_test(compare_test[variable], Measurement)[variable] 
                    # Store results
                    noMeasurements = (int) (len(Measurement))
                    Results[fold, 5:(5+noMeasurements)] = np.array(accuracies)
                    Results[fold,(5+noMeasurements):(7+noMeasurements)] = np.array([effective_epochs, running_time])
                    Results[fold,(7+noMeasurements):] = AUC
                    setting_acc += accuracies 
                    besties_setting = accuracies
                    best_generators_setting = generators
                    best_AUC_setting = AUC
                    
                    # Plot all images
                    if(Test_Plot[fold]==True):
                        for variable in range(len(Impute_Cols)):
                                PlotHists(compare_test[variable], Var_Names, plotcols=1, out_path=out_path, test=True)
                                PRD_Plot(compare_test[variable], Var_Names,plotcols=1, out_path=out_path, test=True)
                                PRD = Image.open(out_path + "PRD_TestSet.png")
                                Hists=Image.open(out_path + "Hists_TestSet.png")
                                Combined=[Hists, PRD]
                                Complete=Progress_Plot(1, 2, Combined, time.time(), fold, Var_Names,  
                                                       name= GAIN_type + ": " + Var_Names[variable] + " plot Test set" + " Using Configuration = " + (str) (Setting),
                                                       test=True)
                                plt.show()
                    print(" ")
                    print("Performance fold:",  accuracies)
                    print("AUC: ", AUC)
                
                
                # Actual CV
                if(CV_Complete==True):
                    np.random.seed(seed)
                    KFold=StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
                    y=np.ones(np.shape(all_data)[0])
                    
                    # the current fold            
                    fold=0
                
                    # Store results
                    Results = np.zeros((Nrows,Ncols))
                    
                    # For every obtain GAIN performances
                    for train_i, test_i in (KFold.split(all_data,y)):
                        # Store hyperparameters in results
                        Results[fold, 0] = batch_size
                        Results[fold, 1] = hint_rate 
                        Results[fold, 2] = lambda_0
                        Results[fold, 3] = lambda_3
                        Results[fold, 4] = seed
                        print("Start training for fold ", fold+1)
                        # Obtain training and test observations
                        train_data=all_data[train_i]
                        test_data=all_data[test_i]
                        train_actual, train_impute, train_mask=data_Imputer(train_data, train_missrate, Impute_Cols, seed=seed)
                        test_actual, test_impute, test_mask=data_Imputer(test_data, test_missrate, Impute_Cols, seed=seed)
                        # Normalise test
                        test_normalised, test_paras= normalization(test_impute)
                        # Generator input 
                        Generator_Input, Embedded_Imputations, Layer_Size, Dim, Norm_Data_Rest, Norm_Pars, Data_Rest, M_Rest = Embedded_GainInput(test_impute, Embed_Cols, 
                                                                                                               Impute_Cols, ceiling, seed=seed )
                        # Select which GAIN method to use
                        if(GAIN_type=="GAIN"):
                            train_gain =gain(train_actual, train_impute, batch_size, hint_rate, lambda_0, epochs=training_epochs, ceiling=ceiling, 
                                           conditioned_hints=True, conditioned_discriminator=conditioned_critic, conditioned_MSE=conditioned_MSE, 
                                           Embed_Cols=Embed_Cols,Var_Names=Var_Names, Plot=Plot_Gain[fold], out_path=out_path, generator_path=generator_path)
                         
                        if(GAIN_type=="WGAIN"):
                            train_gain =wgaingp(train_actual, train_impute, batch_size, hint_rate, lambda_0, lambda_3, epochs=training_epochs, ceiling=ceiling, 
                                           conditioned_hints=True, conditioned_critic=conditioned_critic, conditioned_MSE=conditioned_MSE, 
                                           Embed_Cols=Embed_Cols,Var_Names=Var_Names, Plot=Plot_Gain[fold], out_path=out_path, generator_path=generator_path,
                                           GAIN_type=GAIN_type)
                        if(GAIN_type=="WGAINGP"):
                            train_gain= wgaingp(train_actual, train_impute, batch_size, hint_rate, lambda_0, lambda_3, epochs=training_epochs, ceiling=ceiling, 
                                           conditioned_hints=True, conditioned_critic=conditioned_critic, conditioned_MSE=conditioned_MSE, 
                                           Embed_Cols=Embed_Cols,Var_Names=Var_Names, Plot=Plot_Gain[fold], out_path=out_path, generator_path=generator_path,
                                           GAIN_type=GAIN_type)
                        
                        # Extract generators and obtain compare results
                        generators = []
                        compare_test = []
                        for variable in range(len(Impute_Cols)):
                            generator=train_gain[1][1][variable]
                            generators.append(generator)
                            compare_test.append(Embed_Compare(Generator_Input, Embedded_Imputations, Layer_Size, Dim,
                                             Norm_Data_Rest,  Norm_Pars, Data_Rest, M_Rest, test_actual, test_mask,Impute_Cols,generator))
                        effective_epochs = train_gain[1][2]
                        running_time =train_gain[1][3]
                        AUC=train_gain[1][4]
                        
                        # Compute accuracies 
                        noMeasurements = (int) (len(Measurement))
                        accuracies = np.zeros(len(Impute_Cols))
                        for variable in range(noMeasurements):
                            accuracies[variable]=accuracy_test(compare_test[variable], Measurement)[variable]
                        Results[fold, 5:(5+noMeasurements)] = np.array(accuracies)
                        Results[fold,(5+noMeasurements):(7+noMeasurements)] = np.array([effective_epochs, running_time])
                        Results[fold,(7+noMeasurements):]
                        setting_acc += accuracies *1/K
                        # Check if performance of current fold is best
                        for i in range(len(Measurement)):
                                if(Measurement[i]=="RMSE"):
                                    if(accuracies[i]<besties_setting[i]):
                                        besties_setting[i]=accuracies[i]
                                        best_generators_setting[i]=generators[i]
                                        best_AUC_setting[i]=AUC[i]
                                if(Measurement[i]!="RMSE"):
                                    if(accuracies[i]>besties[i]):
                                        besties_setting[i]=accuracies[i]
                                        best_generators_setting[i]=generators[i]
                                        best_AUC_setting[i]=AUC[i]
      
                        # Plot all images
                        if(Test_Plot[fold]==True):
                            # Plot Histograms and PRD-plot
                            for variable in range(len(Impute_Cols)):
                                PlotHists(compare_test[variable], Var_Names, plotcols=1, out_path=out_path, test=True)
                                PRD_Plot(compare_test[variable], Var_Names,plotcols=1, out_path=out_path, test=True)
                                PRD = Image.open(out_path + "PRD_TestSet.png")
                                Hists=Image.open(out_path + "Hists_TestSet.png")
                                Combined=[Hists, PRD]
                                Complete=Progress_Plot(1, 2, Combined, time.time(), fold, Var_Names,  
                                                       name= GAIN_type + ": " + Var_Names[variable] + " plot Test set" + " Using Configuration = " + (str) (Setting),
                                                       test=True)
                                plt.show()
                        fold+=1
                        print(" ")
                        print("Performance fold:",  accuracies)
                        print("AUC fold:", AUC )
                            
                    print(" ")
                    print("Average performance over all folds: ", setting_acc)
      
                # Save results
                Results= pd.DataFrame(Results)
                Setting_name = (str) (Setting)
                if(len(Measurement)==2):
                    head= ["batch size", "hint rate",  "lambda_0", "lambda_3", "seed", "RMSE", 
                           "F in the chat", "effective epochs", "running time (sec)", "AUC Shipmentdays", "AUC Transportertype"]
                if(len(Measurement)==1):
                   head= ["batch size", "hint rate",  "lambda_0", "lambda_3", "seed", "RMSE", 
                          "effective epochs", "running time (sec)", "AUC Transporter"]
                if(CV_Complete==False):
                    Results.to_csv(CV_path+ GAIN_type + "Test_setting" + Setting_name+".csv", header=head, index=False)
                if(CV_Complete==True):
                    Results.to_csv(CV_path+ GAIN_type + "CV_setting" + Setting_name+".csv", header=head,index=False)
                
                # Update if we have better results under new setting
                for i in range(len(Measurement)):
                        if(Measurement[i]=="RMSE"):
                            if(setting_acc[i]<besties[i]):
                                besties[i]=setting_acc[i]
                                best_generators[i]=best_generators_setting[i]
                                best_AUC[i]=best_AUC_setting[i]
                                best_hypers[i,:]=[batch_size, lambda_0, lambda_3]
                                best_configs[i]=Setting
 
                        if(Measurement[i]!="RMSE"):
                            if(setting_acc[i]>besties[i]):
                                besties[i]=setting_acc[i]
                                best_generators[i]=best_generators_setting[i]
                                best_hypers[i,:]=[batch_size, lambda_0, lambda_3]
                                best_AUC[i]=best_AUC_setting[i]
                                best_configs[i]=Setting
                Setting+=1

    print("Configurations leading to best results: ", best_configs)
    print("Best accuracies over all configurations: ", besties)    
    print("Hyperparameters of best configuration: ")
    print(best_hypers)
    print("AUC of best hypers", best_AUC)
    

    return([best_generators, besties, best_AUC])
