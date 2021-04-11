# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:11:01 2021

@author: Het Gespleten Geweten
"""

# In this file, we split a cleaned dataset into multiple parts. Furthermore, the cleaned dataset will be used
# to compute the probabilities that will be used for determining the binary class outcomes. 
# First, we will find the empirical distributions for all binary class outcomes. For this the entire cleaned
# dataset wil be used. After this is done, we move on to the next part. Here we will try to build the overall model. 
# For this, we will split the overall data into two parts: One part for training the GAINS and one part for
# testing the overall model. For training the GAINS, we will pick 80% of all the data. Hence, 
# we have a pure test set of 20% of the data. Within the data that is used for GAIN training, 
# We need to split dataset into two parts:
    # 1. Part 1 GAIN: Here we want to impute the Shipment Days and Transporter Type. 
    #                 For this part, we can only use the data which was not cancelled. 
    # 2. Part 2 GAIN: Here we want to impute the transporter days. We can only use the data
    #                 which have a known shipment date and trustable transporter name. 

import pandas as pd 
import numpy as np
from tqdm import tqdm

# Own files
# Set here path for data extraction (out_path) and for storing data of empirical distributions (empirical_path)
# output path for storing files
path= "C:/Users/Lars Hurkmans/Downloads/GAIN_CODE"
out_path = path + "/Data/"
# Path for empirical probabilities
empirical_path = path+ "/Empirical/"

print(" ")
print("Preparing data for the crossroads that lie ahead")
print(" ")
all_data= pd.read_csv(out_path+"data_sample_full.csv",low_memory=False)
# Drop unnamed: 0 column
all_data=all_data.drop("Unnamed: 0.1", axis=1)
all_data=all_data.drop("Unnamed: 0", axis=1)

# Drop all orders with cardinality of a level under the treshold
product_type="productGroup"
product_treshold=5
product_names= (all_data[product_type].value_counts()>product_treshold)
product_names=product_names.index[product_names==False]
for product in tqdm(range(len(product_names))):
    all_data=all_data.drop(all_data[all_data[product_type]==product_names[product]].index)
print("Removed", product_type, "levels with cardinality < ", product_treshold)
print(" ")

# code matchings
MatchC0des = all_data["detailedMatchClassification"]
all_data["MatchC0des"]= MatchC0des
MatchC0des = all_data["MatchC0des"].replace({"UNHAPPY":-1,
                                 "UNKNOWN": -2, 
                                 "KNOWN HAPPY":-3}, inplace=True)
all_data['Return'] = np.where(all_data['ReturnDays'] > -1, 0, all_data['ReturnDays'])
all_data['Return'] = all_data['Return'] +1
all_data['Case'] = np.where(all_data['CaseDays'] > -1, 0, all_data['CaseDays'])
all_data['Case'] = all_data['Case'] +1


# Names of columns
names = all_data.head()
names = list(names.columns)

print("Real distribution matches:")
print(all_data["detailedMatchClassification"].value_counts()/np.shape(all_data)[0])
print(" ")
# Function which extracts all empirical distributions, as well as datasets for the two imputation parts
def RoadToGlory(all_data):
    # Extract variable names
    names = all_data.head()
    names = list(names.columns)
    
    # Find the binary variable outcomes. In total we consider 7 binary outcomes:
        # 1. Cancellation --> Unhappy or rest
        # 2. Return conditioned on unknown shipment date and/or untrustable transporter --> Unhappy or Case
        # 3. Case conditioned on unknown shipment date and/or untrustable transporter --> Unhappy or Unknown
        # 4. Return conditioned on unknown delivery date --> Unhappy or case
        # 5. Case conditioned on no return and unknown delivery --> Unhappy or unknown
        # 6. Return conditioned on on time delivery date --> Unhappy or case
        # 7. Case conditioned on no return and on time delivery --> Happy or Unhappy 
    # For all probabilities, the first probabilty is the probability on the Unhappy result.
    # While we update this, we also will remove variables from the dataset. Hence, we will during this
    # process also create the datasets (without training-test splits) for the GAINS.
    # For computing the baseline, we will also compute these probabilities already. 
    
    # 1. Cancellation or no cancellation: We look at column noCancellation
    cancels = all_data["noCancellation"]
    cancels=[np.sum(cancels)/len(cancels), 1-np.sum(cancels)/len(cancels)]   
    
    # Remove all cancellations from data --> first all labeled canceled
    no_cancel_data = all_data.drop(all_data[all_data["noCancellation"] == False].index)
    # Next, we also remove all actual cancellations (cancellation data >-1), but given a lable of true,
    # since these do not have a shipment date (assumption). 
    no_cancel_data=no_cancel_data.drop(no_cancel_data[no_cancel_data["Canceldays"] != -1].index)
    
    # We also remove shipmentdays with -1, since the number of observations that are left over with this situation
    # are quite low and do therefore not contribute much (assumption)
    no_cancel_data=no_cancel_data.drop(no_cancel_data[no_cancel_data["ShipmentDays"] == -1].index)
    
    # Empirical distribution over deliveries 
    deliveries=np.unique(np.array(no_cancel_data["onTimeDelivery"]),return_counts=True)[1]/len(np.array(no_cancel_data["onTimeDelivery"]))
    
    # Data with unknown delivery date
    unknowndel_data=no_cancel_data.copy()
    # Remove returned situations for calculating case probability
    unknowndel_data= unknowndel_data.drop(unknowndel_data[unknowndel_data["onTimeDelivery"] !='unknown'].index)  
    
    # 2. Compute empirical probabilities for return
    return1_data=unknowndel_data.copy()
    return1_data['ReturnDays'] = np.where(return1_data['ReturnDays'] > -1, 0, return1_data['ReturnDays'])
    return1_data['ReturnDays'] = return1_data['ReturnDays'] +1
    return1_data["Return"] = return1_data["ReturnDays"]
    return1 = unknowndel_data["ReturnDays"]    
    return1 = [np.sum(return1>-1)/len(return1), 1- np.sum(return1>-1)/len(return1)]
                
    # 3.  Calculate empirical probabilites for first case moment
    case1_data=unknowndel_data.copy()
    case1_data['CaseDays'] = np.where(case1_data['CaseDays'] > -1, 0, case1_data['CaseDays'])
    case1_data['CaseDays'] = case1_data['CaseDays'] +1
    case1_data["Case"] = case1_data["CaseDays"]
    case1=  unknowndel_data["CaseDays"]    
    case1 = [np.sum(case1>-1)/len(case1), 1- np.sum(case1>-1)/len(case1)]    
    
    # Data with unknown delivery date
    knowndel_data=no_cancel_data.copy()
    # Remove returned situations for calculating case probability
    knowndel_data= knowndel_data.drop(knowndel_data[knowndel_data["onTimeDelivery"] !="true"].index)
            
    # 4. Compute empirical probabilities for return
    return2_data=knowndel_data.copy()
    return2_data['ReturnDays'] = np.where(return2_data['ReturnDays'] > -1, 0, return2_data['ReturnDays'])
    return2_data['ReturnDays'] = return2_data['ReturnDays'] +1
    return2_data["Return"] = return2_data["ReturnDays"]
    return2 = knowndel_data["ReturnDays"]    
    return2 = [np.sum(return2>-1)/len(return2), 1- np.sum(return2>-1)/len(return2)]
                
    # 5.  Calculate empirical probabilites for first case moment
    case2_data=knowndel_data.copy()
    case2_data['CaseDays'] = np.where(case2_data['CaseDays'] > -1, 0, case2_data['CaseDays'])
    case2_data['CaseDays'] = case2_data['CaseDays'] +1
    case2_data["Case"] = case2_data["CaseDays"]
    case2=  knowndel_data["CaseDays"]    
    case2 = [np.sum(case2>-1)/len(case2), 1- np.sum(case2>-1)/len(case2)]  
    
    # Return all the required parts: 
        # 1. cancellation probabilities: first element is no cancel
        # 2. return probabilities: first element is return
        # 3. case probabilities: first element is case
        # 4. Datasets for data imputation no_cancel_data (delivery type)
        #    return_data and case_data
    
    return([cancels, deliveries,
            return1, return2, 
            case1, case2,
            no_cancel_data,
            return1_data, case1_data,
            return2_data, case2_data])

# Convert data to  --> used for GAIN methods
def categoryCreator (all_data, product_type):
# Create variable for 
    all_data[product_type]=all_data[product_type].astype("category")
    all_data[product_type]=all_data[product_type].cat.remove_unused_categories()
    all_data[product_type +"_code"]=all_data[product_type].cat.codes
    all_data["Order_month"]=all_data["Order_month"].astype("category")
    all_data["Order_month_code"]=all_data["Order_month"].cat.codes
    all_data["Order_weekday"]=all_data["Order_weekday"].astype("category")
    all_data["Order_weekday_code"]=all_data["Order_weekday"].cat.codes
    return(all_data)
        
# Extract all data. elements 0-6, 10-12 could here be extracted to
# obtain the empirical probabilities. 
print("Enter the first crossroads")
model_toolkit= RoadToGlory(all_data)    

# empirical distributions, used for base case. 
empirical_cancel = model_toolkit[0]
empirical_delivery= model_toolkit[1]
empirical_return1, empirical_return2= model_toolkit[2:4]
empirical_case1, empirical_case2=   model_toolkit[4:6]

# save empirical distributions in files to load them later
pd.DataFrame(empirical_cancel).to_csv(empirical_path + "cancel.csv", index=False)
pd.DataFrame(empirical_delivery).to_csv(empirical_path + "delivery.csv", index=False)
pd.DataFrame(empirical_return1).to_csv(empirical_path + "return1.csv", index=False)
pd.DataFrame(empirical_return2).to_csv(empirical_path + "return2.csv", index=False)
pd.DataFrame(empirical_case1).to_csv(empirical_path + "case1.csv", index=False)
pd.DataFrame(empirical_case2).to_csv(empirical_path + "case2.csv", index=False)


# Convert data to  --> used for GAIN methods
all_data[product_type]=all_data[product_type].astype("category")
all_data[product_type +"_code"]=all_data[product_type].cat.codes
all_data["Order_month"]=all_data["Order_month"].astype("category")
all_data["Order_month_code"]=all_data["Order_month"].cat.codes
all_data["Order_weekday"]=all_data["Order_weekday"].astype("category")
all_data["Order_weekday_code"]=all_data["Order_weekday"].cat.codes


# Next, we split the data into two parts, one part for training the GAINS and one other
# for testing the overall model tree. This is a stratified sample based on the product type distribution
GAIN_frac=0.33
GAIN_data=all_data.groupby(product_type, group_keys=False).apply(lambda x: x.sample(frac=GAIN_frac, random_state=911))
tree_data = all_data.drop(GAIN_data.index)

# check if we have na's in GAIN data and remove them if this happens.
# Would we not do this, the number of levels for the categorical variables would not be
# correct. 
GAIN_data=GAIN_data.dropna(axis=0)

print("Enter the second crossroads")
print(" ")

# For GAIN data create the data for the two GAIN imputation moments.
GAIN_data= RoadToGlory(GAIN_data)
GAIN_delivery= GAIN_data[-5]
GAIN_return1=GAIN_data[-4]
GAIN_case1=GAIN_data[-3]
GAIN_return2=GAIN_data[-2]
GAIN_case2=GAIN_data[-1]

# Remove product type names which do not occur in one of the tree datasets.
product_names= (GAIN_delivery[product_type].value_counts()>0)
product_names=product_names.index[product_names==False]
for product in range(len(product_names)):
    tree_data=tree_data.drop(tree_data[tree_data[product_type]==product_names[product]].index)
    GAIN_delivery=GAIN_delivery.drop(GAIN_delivery[GAIN_delivery[product_type]==product_names[product]].index)

# Convert categorical data
tree_data = categoryCreator(tree_data, product_type=product_type)
GAIN_delivery = categoryCreator(GAIN_delivery, product_type=product_type)
GAIN_return1 = categoryCreator(GAIN_return1, product_type=product_type)
GAIN_return2 = categoryCreator(GAIN_return2, product_type=product_type)
GAIN_case1 = categoryCreator(GAIN_case1, product_type=product_type)
GAIN_case2 = categoryCreator(GAIN_case2, product_type=product_type)

# Check if all the levels of the product type match with all the datasets
tree_names = np.sort(tree_data[product_type].cat.categories)
GAIN1_names = np.sort(GAIN_delivery[product_type].cat.categories)
return1_names = np.sort(GAIN_return1[product_type].cat.categories)
case1_names = np.sort(GAIN_case1[product_type].cat.categories)
return2_names = np.sort(GAIN_return2[product_type].cat.categories)
case2_names = np.sort(GAIN_case2[product_type].cat.categories)
print("number of", product_type , "levels after cleaning:")
print("tree_data", len(tree_names) )
print("GAIN dataset 1", len(GAIN1_names))
print("GAIN return 1",  len(return1_names))
print("GAIN return 2",  len(return2_names))
print("GAIN case 1",  len(case1_names))
print("GAIN case 2",  len(case2_names))

print("tree_data has same categories as GAIN dataset? --> ", (np.sum(tree_names==GAIN1_names)==len(tree_names)))
print("GAIN return 1 has same categories as GAIN dataset 1? --> ", (np.sum(return1_names==GAIN1_names)==len(tree_names)))
print("GAIN case 1 has same categories as GAIN dataset 1? --> ", (np.sum(case1_names==GAIN1_names)==len(tree_names)) )
print("GAIN return 2 has same categories as GAIN dataset 1? --> ", (np.sum(return2_names==GAIN1_names)==len(tree_names)))
print("GAIN case 2 has same categories as GAIN dataset 1? --> ", (np.sum(case2_names==GAIN1_names)==len(tree_names)) )
tree_names=np.sort(tree_data[product_type].value_counts().index)
GAIN1_names=np.sort(GAIN_delivery[product_type].value_counts().index)



# For the two GAIN datasets, we now make them appropiate for being complacent with
# the GAIN models. For this, we first need to decide which variables we want to use. 
# Drop these variables for first GAIN dataset
dropcols= ["orderDate", "sellerId", "cancellationReasonCode", 
            "quanityReturned", "countryOriginSeller","calculationDefinitive",
            "noCancellation", "generalMatchClassification",
            "detailedMatchClassification", "Canceldays",
            "ReturnDays", "CaseDays",
            "detailedMatchClassification","Belgium", "2020", "FBB" ,
            product_type, "transporterName", "Order_month", 
            "Order_weekday", "MatchC0des", "ID",
            "DeliveryDays", "Return", "Case",
            "ShipmentDays","TransporterDays"]


returndrop= ["orderDate", "sellerId", "cancellationReasonCode", 
            "quanityReturned", "countryOriginSeller", "calculationDefinitive",
            "noCancellation", "onTimeDelivery", "generalMatchClassification",
            "detailedMatchClassification", "Canceldays",
            "ReturnDays", "CaseDays",
            "detailedMatchClassification","Belgium", "2020", "FBB" ,
            product_type, "transporterName", "Order_month", 
            "Order_weekday", "MatchC0des", "ID",
            "DeliveryDays", "TransporterDays",
            "ShipmentDays","Case"]



casedrop= ["orderDate", "sellerId", "cancellationReasonCode", 
            "quanityReturned", "countryOriginSeller", "calculationDefinitive",
            "noCancellation", "onTimeDelivery", "generalMatchClassification",
            "detailedMatchClassification", "Canceldays",
            "ReturnDays", "CaseDays",
            "detailedMatchClassification","Belgium", "2020", "FBB" ,
            product_type, "transporterName", "Order_month", 
            "Order_weekday", "MatchC0des", "ID",
            "DeliveryDays", "TransporterDays", 
            "ShipmentDays", "Return"]

return1drop= ["orderDate", "sellerId", "cancellationReasonCode", 
            "quanityReturned", "countryOriginSeller", "calculationDefinitive",
            "noCancellation", "onTimeDelivery", "generalMatchClassification",
            "detailedMatchClassification", "Canceldays",
            "ReturnDays", "CaseDays",
            "detailedMatchClassification","Belgium", "2020", "FBB" ,
            product_type, "transporterName", "Order_month", 
            "Order_weekday", "MatchC0des", "ID",
            "DeliveryDays", "TransporterDays",
            "ShipmentDays","Case", product_type +"_code"]



case1drop= ["orderDate", "sellerId", "cancellationReasonCode", 
            "quanityReturned", "countryOriginSeller", "calculationDefinitive",
            "noCancellation", "onTimeDelivery", "generalMatchClassification",
            "detailedMatchClassification", "Canceldays",
            "ReturnDays", "CaseDays",
            "detailedMatchClassification","Belgium", "2020", "FBB" ,
            product_type, "transporterName", "Order_month", 
            "Order_weekday", "MatchC0des", "ID",
            "DeliveryDays", "TransporterDays", 
            "ShipmentDays", "Return", product_type +"_code"]


GAIN_delivery = GAIN_delivery.drop(dropcols, axis=1) 
GAIN_return1=GAIN_return1.drop(return1drop, axis=1) 
GAIN_case1 = GAIN_case1.drop(case1drop, axis=1) 
GAIN_return2=GAIN_return2.drop(returndrop, axis=1) 
GAIN_case2 = GAIN_case2.drop(casedrop, axis=1) 


# Convert on time delivery to numeric
tree_data["onTimeDelivery"]=tree_data["onTimeDelivery"].astype("category")
tree_data["onTimeDelivery"]=tree_data["onTimeDelivery"].cat.codes
GAIN_delivery["onTimeDelivery"]=GAIN_delivery["onTimeDelivery"].astype("category")
GAIN_delivery["onTimeDelivery"]=GAIN_delivery["onTimeDelivery"].cat.codes

# Store data as csv, such that we can load it seperately in GAINhub
GAIN_delivery.to_csv(out_path+ "GAIN1.csv", index=False)
GAIN_return1.to_csv(out_path+ "GAIN_return1.csv", index=False)
GAIN_case1.to_csv(out_path+ "GAIN_case1.csv", index=False)
GAIN_return2.to_csv(out_path+ "GAIN_return2.csv", index=False)
GAIN_case2.to_csv(out_path+ "GAIN_case2.csv", index=False)
tree_data.to_csv(out_path+ "WayOutTree.csv", index=False)


# Baseline probability accuracies

# Baseline probability accuracies
N=len(np.array(all_data["noCancellation"]))
Baseline_cancel = np.sum(np.random.choice(2,N,p=(1-np.array(empirical_cancel)))==np.array(all_data["noCancellation"]))/N

N=len(np.array(GAIN_delivery["onTimeDelivery"]))
Baseline_delivery = np.sum(np.random.choice(3,N,p=(np.array(empirical_delivery)))==np.array(GAIN_delivery["onTimeDelivery"]))/N

N1= len(np.array(GAIN_return1["Return"]))
Baseline_return1 = np.sum(np.random.choice(2,N1,p=(1-np.array(empirical_return1)))==np.array(GAIN_return1["Return"]))/N1
Baseline_case1 = np.sum(np.random.choice(2,N1,p=(1-np.array(empirical_case1)))==np.array(GAIN_case1["Case"]))/N1

N2= len(np.array(GAIN_return2["Return"]))
Baseline_return2 = np.sum(np.random.choice(2,N2,p=(1-np.array(empirical_return2)))==np.array(GAIN_return2["Return"]))/N2
Baseline_case2 = np.sum(np.random.choice(2,N2,p=(1-np.array(empirical_case2)))==np.array(GAIN_case2["Case"]))/N2























    