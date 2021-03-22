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
    
    # Find names of transporters which always have unknown deliveries (0.98 assumption)
    unknown_treshold = 0.96
    transporter_rep= pd.crosstab(no_cancel_data["transporterName"],no_cancel_data["onTimeDelivery"])
    no_transporters=np.shape(transporter_rep)[0]
    # totals will be used to compute the baseline probabilities for every transporter based on the 
    # number of occurences. 
    totals = np.zeros(no_transporters)
    for transporter in range(no_transporters):
        totals[transporter] = np.sum(np.array(transporter_rep[transporter:transporter+1]))
        transporter_rep[transporter:transporter+1]= transporter_rep[transporter:transporter+1]/np.sum(np.array(transporter_rep[transporter:transporter+1]))
    
    totals= totals/np.sum(totals)
    # index corresponding to the transporter names. used for tracking which probability belongs
    # to which transporter
    transporter_names = transporter_rep.index
    transporter_probs = pd.DataFrame(totals,index=transporter_names)
    
    # For Baseline, also compute empirical distribution of shipment days (threshold 8 days assumption)
    shipmentdays = no_cancel_data["ShipmentDays"].value_counts().sort_index() 
    shipmentdays = shipmentdays/np.sum(shipmentdays)
    
    # Get names of transporters which always have unknown deliveries and the rest
    unknown_transporters = np.array(transporter_rep)
    unknown_transporters= np.where(unknown_transporters[:,-1]>unknown_treshold)
    known_transporters= np.arange(0,no_transporters)
    known_transporters= known_transporters[~np.in1d(known_transporters,unknown_transporters).reshape(known_transporters.shape)]
    unknown_transporters = transporter_rep.index[unknown_transporters]
    known_transporters = transporter_rep.index[known_transporters]
    
    # Split data again: This time, remove all unknown shipment dates and transporter names for second GAIN
    # And use the removed part to compute binary probabilities for (2)
    # with delivery date unknown
    # Data for first case
    unknown_data= no_cancel_data
    for remove in range(len(known_transporters)):
        unknown_data=unknown_data.drop(unknown_data[unknown_data["transporterName"] == known_transporters[remove]].index)
        
    # We first calculate the probabilities for return and case based on the unknown_data. 
    # For this, we first need to remove some inconsistencies. For example, we cannot have 
    # any happy matchings. 
    unknown_data= unknown_data.drop(unknown_data[unknown_data["detailedMatchClassification"] == "KNOWN HAPPY"].index)
        
    # 2. Compute empirical probabilities for return
    return1 = unknown_data["ReturnDays"]    
    return1 = [np.sum(return1>-1)/len(return1), 1- np.sum(return1>-1)/len(return1)]
        
    # Remove returned situations for calculating case probability
    unknown_data= unknown_data.drop(unknown_data[unknown_data["ReturnDays"] > -1].index)    
        
    # 3.  Calculate empirical probabilites for first case moment
    case1=  unknown_data["CaseDays"]    
    case1 = [np.sum(case1>-1)/len(case1), 1- np.sum(case1>-1)/len(case1)]    
        
    # Using the data from the second GAIN. we are now left with calculating more binary probabilities
    # conditioned on data. 
    # Data for second GAIN
    delivery_data= no_cancel_data
    for remove in range(len(unknown_transporters)):
        delivery_data=delivery_data.drop(delivery_data[delivery_data["transporterName"] == unknown_transporters[remove]].index)
    
    # For delivery data check that we do not have any cancelled orders left
    delivery_data=delivery_data.drop(delivery_data[delivery_data["DeliveryDays"] ==-2].index)
              
    # Compute empirical distribution for delivery days
    transporterdays=delivery_data["TransporterDays"].value_counts().sort_index() 
    transporterdays= transporterdays/np.sum(transporterdays)
    
    # Obtain data condtioned on unknown delivery
    unknowndel_data = delivery_data
    unknowndel_data=unknowndel_data.drop(unknowndel_data[unknowndel_data["TransporterDays"]!=-1].index)
    
    # 4.  Calculate emprical return probabilities for unknown deliveries (due to TransporterDays)
    return2 = unknowndel_data["ReturnDays"]    
    return2 = [np.sum(return2>-1)/len(return2), 1- np.sum(return2>-1)/len(return2)]
        
    # Remove returned situations for calculating case probability
    unknowndel_data= unknowndel_data.drop(unknowndel_data[unknowndel_data["ReturnDays"] > -1].index)       
        
    # 5. Calculate empirical probabilites for second case moment
    case2=  unknowndel_data["CaseDays"]    
    case2 = [np.sum(case2>-1)/len(case2), 1- np.sum(case2>-1)/len(case2)] 
         
    # Next remove all data with unknown/late deliveries, such that return and cases for 
    # on time deliveries can be calculated
    
    timedel_data=  delivery_data
    timedel_data= timedel_data.drop(timedel_data[timedel_data["onTimeDelivery"]!= "true"  ].index)

    # 6.  Calculate emprical return probabilities for known deliveries (due to TransporterDays)
    return3 = timedel_data["ReturnDays"]    
    return3 = [np.sum(return3>-1)/len(return3), 1- np.sum(return3>-1)/len(return3)]
        
    # Remove returned situations for calculating case probability
    timedel_data= timedel_data.drop(timedel_data[timedel_data["ReturnDays"] > -1].index)       
        
    # 7. Calculate empirical probabilites for second case moment
    case3=  timedel_data["CaseDays"]    
    case3 = [np.sum(case3>-1)/len(case3), 1- np.sum(case3>-1)/len(case3)] 
    
    
    # Return all the required parts: 
        # 1. cancellation probabilities: first element is no cancel
        # 2. return probabilities: first element is return
        # 3. case probabilities: first element is case
        # 4. transporter reputation, as well as the names of 
        #    the unknown transporters known transporters and 
        #    their empirical probabilities of occuring
        # 5. emprical distributions for shipment days and transporter days
        # 6. Datasets for data imputation no_cancel_data (transporter name and shipment days)
        #    and delivery_data (transporter days)
    
    return([cancels, 
            return1, return2, return3, 
            case1, case2, case3,
            transporter_rep, unknown_transporters, known_transporters, totals,
            shipmentdays, transporterdays,
            no_cancel_data, delivery_data])
# Convert data to  --> used for GAIN methods
def categoryCreator (all_data, product_type):
# Create variable for 
    all_data[product_type]=all_data[product_type].astype("category")
    all_data[product_type]=all_data[product_type].cat.remove_unused_categories()
    all_data[product_type +"_code"]=all_data[product_type].cat.codes
    all_data["transporterName"]=all_data["transporterName"].astype(ordered_transcats)
    all_data["transporterName_code"]=all_data["transporterName"].cat.codes
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
empirical_return1, empirical_return2 , empirical_return3  = model_toolkit[1:4]
empirical_case1, empirical_case2, empirical_case3=   model_toolkit[4:7]
empirical_transporter_reputations = model_toolkit[7]
empirical_transporters, empirical_shipmentdays, empirical_transporterdays= model_toolkit[10:13]

# save empirical distributions in files to load them later
pd.DataFrame(empirical_cancel).to_csv(empirical_path + "cancel.csv", index=False)
pd.DataFrame(empirical_return1).to_csv(empirical_path + "return1.csv", index=False)
pd.DataFrame(empirical_return2).to_csv(empirical_path + "return2.csv", index=False)
pd.DataFrame(empirical_return3).to_csv(empirical_path + "return3.csv", index=False)
pd.DataFrame(empirical_case1).to_csv(empirical_path + "case1.csv", index=False)
pd.DataFrame(empirical_case2).to_csv(empirical_path + "case2.csv", index=False)
pd.DataFrame(empirical_case3).to_csv(empirical_path + "case3.csv", index=False)
pd.DataFrame(empirical_shipmentdays).to_csv(empirical_path + "shipment.csv", index=False)
pd.DataFrame(empirical_transporterdays).to_csv(empirical_path + "transdays.csv", index=False)

# transporter names
transporter_names=np.array(all_data["transporterName"].unique())
# Remove nan
nan_where = np.where(pd.isnull(transporter_names))[0][0]
transporter_names=np.delete(transporter_names,nan_where)
transporter_names= np.delete(transporter_names, np.where(transporter_names=="NONE")[0][0])
transporter_names = np.sort(transporter_names)

# two types of transporters
unknown_transporters=model_toolkit[8]
known_transporters=model_toolkit[9]

# Save transporternames
pd.DataFrame(known_transporters).to_csv(empirical_path + "knownnames.csv", index=False)
pd.DataFrame(unknown_transporters).to_csv(empirical_path + "unknownnames.csv", index=False)

# Combine them in ordered fashion --> Needed for converting to categorical variables
# For the GAINs, we need to make split between known and unknown transporter, in which
# the unknown transporters are removed for the second GAIN dataset. Hence, we first add
# the known transporternames 
ordered_transporternames = np.hstack((known_transporters,unknown_transporters))
ordered_transcats= pd.CategoricalDtype(categories=ordered_transporternames, ordered=True)

# Create binary probabiltities based on transporter type. 
empirical_unknown = 0
for name in range(len(unknown_transporters)):
    element=np.where(transporter_names==unknown_transporters[name])
    empirical_unknown += empirical_transporters[element]

empirical_unknown = np.reshape(np.array([empirical_unknown,1-empirical_unknown]),2)

# Save unknown distribution
pd.DataFrame(empirical_unknown).to_csv(empirical_path + "unknown.csv", index=False)

unknown_transporter=all_data["transporterName"]
all_data["unknownTransporter"] = unknown_transporter
all_data["unknownTransporter"]=all_data["unknownTransporter"].isin(unknown_transporters)
all_data["unknownTransporter"]=all_data["unknownTransporter"].replace({True:1, False:0})


# Convert data to  --> used for GAIN methods
all_data[product_type]=all_data[product_type].astype("category")
all_data[product_type +"_code"]=all_data[product_type].cat.codes
all_data["transporterName"]=all_data["transporterName"].astype(ordered_transcats)
all_data["transporterName_code"]=all_data["transporterName"].cat.codes
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
GAIN_data1= GAIN_data[-2]
GAIN_data2= GAIN_data[-1]

# Remove product type names which do not occur in one of the tree datasets.
product_names= (GAIN_data2[product_type].value_counts()>0)
product_names=product_names.index[product_names==False]
for product in range(len(product_names)):
    tree_data=tree_data.drop(tree_data[tree_data[product_type]==product_names[product]].index)
    GAIN_data1=GAIN_data1.drop(GAIN_data1[GAIN_data1[product_type]==product_names[product]].index)
    GAIN_data2= GAIN_data2.drop( GAIN_data2[ GAIN_data2[product_type]==product_names[product]].index)

# Convert categorical data
tree_data = categoryCreator(tree_data, product_type=product_type)
GAIN_data1 = categoryCreator(GAIN_data1, product_type=product_type)
GAIN_data2 = categoryCreator(GAIN_data1, product_type=product_type)

# Check if all the levels of the product type match with all the datasets
tree_names = np.sort(tree_data[product_type].cat.categories)
GAIN1_names = np.sort(GAIN_data1[product_type].cat.categories)
GAIN2_names = np.sort(GAIN_data2[product_type].cat.categories)
print("number of", product_type , "levels after cleaning:")
print("tree_data", len(tree_names) )
print("GAIN dataset 1", len(GAIN1_names))
print("GAIN dataset 2",  len(GAIN2_names))

print("tree_data has same categories as GAIN dataset2? --> ", (np.sum(tree_names==GAIN2_names)==len(tree_names)) )
print("GAIN dataset 1 has same categories as GAIN dataset2? --> ", (np.sum(GAIN1_names==GAIN2_names)==len(tree_names)) )

tree_names=np.sort(tree_data[product_type].value_counts().index)
GAIN1_names=np.sort(GAIN_data1[product_type].value_counts().index)
GAIN2_names=np.sort(GAIN_data2[product_type].value_counts().index)

all_names=np.transpose(np.vstack((tree_names,GAIN1_names, GAIN2_names)))


# For the two GAIN datasets, we now make them appropiate for being complacent with
# the GAIN models. For this, we first need to decide which variables we want to use. 
# Drop these variables for first GAIN dataset
dropcols1= ["orderDate", "sellerId", "cancellationReasonCode", 
            "quanityReturned", "countryOriginSeller", "calculationDefinitive",
            "noCancellation", "onTimeDelivery", "generalMatchClassification",
            "detailedMatchClassification", "Canceldays",
            "ReturnDays", "CaseDays","TransporterDays",
            "detailedMatchClassification","Belgium", "2020", "FBB",
            product_type, "transporterName", "Order_month", 
            "Order_weekday","transporterName_code", "MatchC0des", "ID",
            "DeliveryDays"]

# Drop these variables for second dataset
dropcols2= ["orderDate", "sellerId", "cancellationReasonCode", 
            "quanityReturned", "countryOriginSeller", "calculationDefinitive",
            "noCancellation", "onTimeDelivery", "generalMatchClassification",
            "detailedMatchClassification", "Canceldays",
            "ReturnDays", "CaseDays",
            "detailedMatchClassification","Belgium", "2020", "FBB" ,
            product_type, "transporterName", "Order_month", 
            "Order_weekday", "MatchC0des", "unknownTransporter", "ID",
            "DeliveryDays", "transporterName_code"]

GAIN_data1 = GAIN_data1.drop(dropcols1, axis=1) 
GAIN_data2 = GAIN_data2.drop(dropcols2, axis=1) 

# Store data as csv, such that we can load it seperately in GAINhub
GAIN_data1.to_csv(out_path+ "GAIN1.csv", index=False)
GAIN_data2.to_csv(out_path+ "GAIN2.csv", index=False)
tree_data.to_csv(out_path+ "WayOutTree.csv", index=False)





























    