# -*- coding: utf-8 -*-
"""
Created on 0 days after the order date

@author: Chriet Titulaer
"""

# In this file, we run the main model tree. We provide 4 different specifications:
    # 1. Baseline: Results based on multinomial sampling
    # 2. GAIN: Results based on original GAIN
    # 3. WGAIN: Results based on WGAIN
    # 4. WGAIN-GP: Results based WGAIN-GP
# In this file, we run two versions of the trees, one based on the generators of the main text
# and onther based on the generators of the appendix. The results of the main text can be found by.
# By running this code you are first asked to specify how many days you want to look after the order date.
# Based on this value, the scores are then computed for the versions. 
# Additionally, you are asked to plot the results for the first 10 days. Note that this can take quite some time. 

# NOTE: The results deviate possibly by a couple of percentage points from the results of the paper, since we changed the code
#       in the R-file after the deadline of the paper (datapreprocessing). Consequently, not the exact same dataset is used 
#       now under the same seed.
#       Additionally, we found a mistake in the printing of the precision of the unhappy class. Initiially, the precision of the
#       unhappy class showed the precision of the happy class. As a result, in the paper, the precision of the unhappy class yields 
#       the same value as the precision of the happy class. Since we found this mistake after the deadline, the precision and F-scores
#       of the unhappy class are therefore different in the outputs of this file than in the paper. 

# HERE STARTS CODE
# Paths for extracting embedded data and generators
# Path (Until map GAIN_CODE)
path= "C:/Users/Lars Hurkmans/Downloads/GAIN_CODE"
out_path = path + "/Data/"
generator_path= path + "/Generators/"
# Path for empirical probabilities
empirical_path = path + "/Empirical/"


import pandas as pd 
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt 

# Own files
from EmbedPret import Embedded_GainInput
from data_loader import data_Imputer


# Load Tree Data
tree_data= pd.read_csv(out_path+"WayOutTree.csv",low_memory=False)

# Load empirical distributions
empirical_cancel= np.array(pd.read_csv(empirical_path + "cancel.csv")).flatten()
empirical_return1= np.array(pd.read_csv(empirical_path + "return1.csv")).flatten()
empirical_return2= np.array(pd.read_csv(empirical_path + "return2.csv")).flatten()
empirical_return3= np.array(pd.read_csv(empirical_path + "return3.csv")).flatten()
empirical_case1= np.array(pd.read_csv(empirical_path + "case1.csv")).flatten()
empirical_case2= np.array(pd.read_csv(empirical_path + "case2.csv")).flatten()
empirical_case3=  np.array(pd.read_csv(empirical_path + "case3.csv")).flatten()
empirical_shipmentdays= np.array(pd.read_csv(empirical_path + "shipment.csv")).flatten()
empirical_transporterdays= np.array(pd.read_csv(empirical_path + "transdays.csv")).flatten()
empirical_unknown= np.array(pd.read_csv(empirical_path + "unknown.csv")).flatten()

# Load transporternames
known_transporters= np.array(pd.read_csv(empirical_path + "knownnames.csv")).flatten()
unknown_transporters= np.array(pd.read_csv(empirical_path + "unknownnames.csv")).flatten()

# Product type used
product_type="productGroup"

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

# Here we create the tree model: This is based on an adjacency list
# This tree will used to track where the order ends up
IndexTree = [
    [2,1],              # Node 0  --> Order --> Node 1 or 2
    [-1],               # Node 1  --> UNHAPPY: cancel --> END NODE (-1)
    [3,4],              # Node 2  --> Imputation of shipment days and transporter name --> Node 3 or 4
    [5,6],              # Node 3  --> Unknown transporter --> Node 5 or 6
    [9,10,11],          # Node 4  --> Known transporter: Impute transporter days --> Node 9, 10 or 11
    [-1],               # Node 5  --> UNHAPPY: Unknown transporter & return --> END NODE (-1)
    [7,8],              # Node 6  --> Unknown transporter & no return --> Node 7 or 8
    [-1],               # Node 7  --> UNHAPPY: Unknown transporter & case --> END NODE (-1)
    [-2],               # Node 8  --> UNKNOWN: Unknown transporter & no return or case --> END NODE (-2)
    [-1],               # Node 9  --> UNHAPPY: Late delivery --> END NODE (-1)
    [12,13],            # Node 10 --> Unknown delivery --> Node 12 or 13
    [16,17],            # Node 11 --> On time delivery --> Node 16 or 17
    [-1],               # Node 12 --> UNHAPPY: Unknown delivery & return --> END Node (-1)
    [14,15],            # Node 13 --> Unknown delivery & no return --> Node 13 or 14
    [-1],               # Node 14 --> UNHAPPY: Unknown delivery & case --> END NODE (-1)
    [-2],               # Node 15 --> UNKNOWN: Unknown delivery & no return or case --> END NODE (-2)
    [-1],               # Node 16 --> UNHAPPY: On time delivery & return --> END NODE(-1)
    [18,19],            # Node 17 --> On time delivery & no return --> Node 18 or 19
    [-1],               # Node 18 --> UNHAPPY: On time delivery & case --> END NODE (-1)
    [-3]]               # Node 19 --> HAPPY --> END NODE (-3)


# Baseline tree model. Here we use multinomial sampling in all nodes.  
def TreeModel(tree_data, currentday=0,seed=123):
    # Set seed
    np.random.seed(seed)
    # The baseline tree
    # match prediction nodes:
    # -1 = UNHAPPY
    # -2 = UNKNOWN
    # -3 = HAPPY
    Baseline = [
        empirical_cancel,                                   # Node 0  --> Coin flip: cancel or no cancel --> Node 1 or 2
        [-1],                                               # Node 1  --> UNHAPPY: cancel --> END NODE (-1)
        [empirical_unknown,empirical_shipmentdays],         # Node 2  --> Coin flip: Shipment days & transporter names
        empirical_return1,                                  # Node 3  --> Coin flip: return 1 --> Node 5 or 6
        empirical_transporterdays,                          # Node 4  --> Coin flip Transporter days
        [-1],                                               # Node 5  --> UNHAPPY: Unknown transporter & return --> END NODE (-1)
        empirical_case1,                                    # Node 6  --> Coin flip: case 1  --> Node 7 or 8
        [-1],                                               # Node 7  --> UNHAPPY: Unknown transporter & case --> END NODE (-1)
        [-2],                                               # Node 8  --> UNKNOWN: Unknown transporter & no return or case --> END NODE (-2)
        [-1],                                               # Node 9  --> UNHAPPY: Late delivery --> END NODE (-1)
        empirical_return2,                                  # Node 10 --> Coin flip: return 2 --> Node 12 or 13
        empirical_return3,                                  # Node 11 --> Coin flip: return 3 --> Node 16 or 17
        [-1],                                               # Node 12 --> UNHAPPY: Unknown delivery & return --> END Node (-1)
        empirical_case2,                                    # Node 13 --> Coin flip: case 2 --> Node 13 or 14
        [-1],                                               # Node 14 --> UNHAPPY: Unknown delivery & case --> END NODE (-1)
        [-2],                                               # Node 15 --> UNKNOWN: Unknown delivery & no return or case --> END NODE (-2)
        [-1],                                               # Node 16 --> UNHAPPY: On time delivery & return --> END NODE(-1)
        empirical_case3,                                    # Node 17 --> Coin flip: case 3 --> Node 18 or 19
        [-1],                                               # Node 18 --> UNHAPPY: On time delivery & case --> END NODE (-1)
        [-3]]                                               # Node 19 --> HAPPY --> END NODE (-3)
    
    # Number of orders of tree data
    obs = np.shape(tree_data)[0]
    predict_nodes = np.zeros(obs)
    # Track the node the order ends up in
    end_nodes = np.zeros(obs)
    
    # Track whether we had knowledge
    Knowledge_is_power= np.full(obs,-1)
    
    # predict matching for every order in tree data
    for order in tqdm(range(obs)):
    
        # Check if our current day gives already information about lagged order data
        # True data
        promiseddays = (int) (tree_data["Promised_Days"].iloc[order])
        actual_canceldays = (int) (tree_data["Canceldays"].iloc[order])
        actual_shipdays = (int) (tree_data["ShipmentDays"].iloc[order])
        actual_transdays = (int) (tree_data["TransporterDays"].iloc[order])
        actual_returndays = (int) (tree_data["ReturnDays"].iloc[order])
        actual_casedays = (int) (tree_data["CaseDays"].iloc[order])
        
        # Store true data --> based on ordering of nodes:
            # 0: canceldays
            # 2: shipmentdays
            # 3: return
            # 4: transporterdays
            # 6: case
            # 10: return
            # 11: return
            # 13: case
            # 17: case
        daynodes= np.array([actual_canceldays, actual_shipdays, 
                            actual_returndays,actual_transdays,
                            actual_casedays,actual_returndays,
                            actual_returndays,actual_casedays,
                            actual_casedays])
        
        predict_days=np.full(len(daynodes),-4)
        
        # Array of nodes which we could skip if we would have the information
        skip_node_indices = np.array([0,2,3,4,6,10,11,13,17])
        skip_nodes=np.full(len(daynodes),-1)
        
        # Check which nodes could be skipped
        for skip_node in range(len(skip_nodes)):
             if(currentday>daynodes[skip_node] and daynodes[skip_node]>=0 ):
                 skip_nodes[skip_node]=1
             
        # Mark the values which we can use. 
        predict_days[np.where(skip_nodes==1)]=daynodes[np.where(skip_nodes==1)]
             
        
        predict_days[np.where(skip_nodes==1)]=daynodes[np.where(skip_nodes==1)]
        
        # Extract actual values of shipmentdays and transporterdays (if known)
        predicted_Shipmentdays= predict_days[1]
        predicted_Transporterdays = predict_days[3]
        predicted_transportername= None
        transporter_element = -1
        
        
        # If we can use actual shipment days, change predictions
        if(predicted_Shipmentdays>=0):
            transporter_element = np.sum(tree_data["transporterName"].iloc[order]==known_transporters)
        
       # If we can use actual transporter days, change predictions
        if(predicted_Transporterdays>=0):
            skip_nodes[0]=1
        
        # Only use nodes we are able to skip
        skip_node_indices = skip_node_indices[skip_nodes>0]            
        
        # Update if we used actual data
        if(len(skip_node_indices)>0):
            if(np.sum(skip_node_indices==2)==1):
                Knowledge_is_power[order]=2
            if(np.sum(skip_node_indices==4)==1):
                Knowledge_is_power[order]=4
        
        # Nodes to keep progression of tree:
        # InNode= current node
        # outNode= new node
        inNode=0
        outNode=0
        
        # Loop through tree to predict matching
        while(outNode>=0):
           inNode=outNode
           element=0
           # Here we predict outcomes. We enter this part if we do not have any 
           # information about the variable concerning the current in coming node
           if(np.sum(inNode==skip_node_indices)!=1):
               # Check that we are not at end node
               if(len(Baseline[inNode])>1):
                # For cancel, case and return, we use multinomial sampling and obtain 
                # results immediately
                if(inNode!=2):   
                    element= np.array(np.where(np.random.multinomial(1,(Baseline[inNode]))==1))[0][0]
                # For shipment days and transporter name (type), we predict two variables
                if(inNode==2):
                    predicted_Shipmentdays =np.array(np.where(Baseline[inNode][1]==Baseline[inNode][1][element]))[0][0]          
                    element= np.array(np.where(np.random.multinomial(1,(Baseline[inNode][0]))==1))[0][0]
                # For transporter days, we check if we have later deilvery    
                if(inNode==4):
                    # 6 days not in training sample
                    if(element==6):
                        element=5
                    predicted_Transporterdays= element-1
                    element=1
                    if(predicted_Transporterdays>=0):
                        element=0
                        predicted_Deliverydays = predicted_Shipmentdays + predicted_Transporterdays
                        if(predicted_Deliverydays <= promiseddays ):
                            element=2
        
           # If we have information about the variable concerning the current node,
           # we enter here
           if(np.sum(inNode==skip_node_indices)==1):
                if(inNode!=2):   
                    element= 0
                if(inNode==2):          
                    element= transporter_element
                if(inNode==4):
                    element=1
                    if(predicted_Transporterdays>=0):
                        element=0
                        random_day = np.random.randint(-1,1)
                        predicted_Deliverydays = predicted_Shipmentdays + predicted_Transporterdays + random_day
                        if(predicted_Deliverydays <= promiseddays ):
                            element=2
                    
            
           outNode=IndexTree[inNode][element]
        predict_nodes[order]=outNode
        end_nodes[order]= inNode
    return((predict_nodes, end_nodes, Knowledge_is_power))

# Trees of GAIN
def GAIN_Model(tree_data, GAIN_type, model="Advanced", currentday=0, seed=123):
    # Set seed
    np.random.seed(seed)
    # Extract the generators to impute the lagged data
    shipment_gain= tf.keras.models.load_model(generator_path+ model +"/" +GAIN_type+"/Ship", compile=False)
    transname_gain= tf.keras.models.load_model(generator_path + model +"/" +GAIN_type+"/Tranname", compile=False)
    transday_gain = tf.keras.models.load_model(generator_path + model + "/" +GAIN_type+"/Transday", compile=False)
    
    # Get gain data for tree. Needed for normalisation of parameters
    gain1_tree=np.array(tree_data.drop(dropcols1,axis=1))
    gain2_tree= np.array(tree_data.drop(dropcols2,axis=1))

    
    embed_cols=[5,16,17,18]
    impute_cols1=[7,15]
    impute_cols2=[9]
    
    tree1, impute1, mask1 = data_Imputer(gain1_tree, 0.99, impute_cols1)
    tree2, impute2, mask2 = data_Imputer(gain2_tree, 0.99, impute_cols2)
    
    
    # Initialise imputation variables
    predicted_Shipmentdays=None
    predicted_transname=None
    predicted_Transporterdays=None 
    
    # GAIN model
    # match prediction nodes:
        # -1 = UNHAPPY
        # -2 = UNKNOWN
        # -3 = HAPPY
    GAIN_Tree = [
        empirical_cancel,                                       # Node 0  --> Coin flip: cancel or no cancel
        [-1],                                                   # Node 1  --> UNHAPPY: cancel --> END NODE (-1)
        [predicted_transname,predicted_Shipmentdays],           # Node 2  --> GAIN 1: Shipment days & transporter name (type) --> Node 3 or 4
        empirical_return1,                                      # Node 3  --> Coin flip: return 1 --> Node 5 or 6
        [predicted_Transporterdays,predicted_Shipmentdays],     # Node 4  --> GAIN 2: Transporter days --> Node 9, 10 or 11
        [-1],                                                   # Node 5  --> UNHAPPY: Unknown transporter & return --> END NODE (-1)
        empirical_case1,                                        # Node 6  --> Coin flip: case 1 --> Node 7 or 8
        [-1],                                                   # Node 7  --> UNHAPPY: Unknown transporter & case --> END NODE (-1)
        [-2],                                                   # Node 8  --> UNKNOWN: Unknown transporter & no return or case --> END NODE (-2)
        [-1],                                                   # Node 9  --> UNHAPPY: Late delivery --> END NODE (-1)
        empirical_return2,                                      # Node 10 --> Coin flip: return 2 --> Node 12 or 13
        empirical_return3,                                      # Node 11 --> Coin flip: return 3 --> Node 16 or 17
        [-1],                                                   # Node 12 --> UNHAPPY: Unknown delivery & return --> END Node (-1)
        empirical_case2,                                        # Node 13 --> Coin flip: case 2 --> Node 14 or 15
        [-1],                                                   # Node 14 --> UNHAPPY: Unknown delivery & case --> END NODE (-1)
        [-2],                                                   # Node 15 --> UNKNOWN: Unknown delivery & no return or case --> END NODE (-2)
        [-1],                                                   # Node 16 --> UNHAPPY: On time delivery & return --> END NODE(-1)
        empirical_case3,                                        # Node 17 --> Coin flip: case 3 --> Node 18 or 19
        [-1],                                                   # Node 18 --> UNHAPPY: On time delivery & case --> END NODE (-1)
        [-3]]                                                   # Node 19 --> HAPPY --> END NODE (-3)
    
    # Minimum and maximum values of imputecolums
    min1 = tree1.min(0)[impute_cols1]
    max1= tree1.max(0)[impute_cols1]
    
    min2 = -1
    max2= tree2.max(0)[impute_cols2]
    
    ceiling=[1,1,1,1]
    
    Gen1 = Embedded_GainInput(impute1, embed_cols, impute_cols1, ceiling, seed=seed )
    Generator1_Input=Gen1[0]
    Embedded1_Imputations=Gen1[1]
    
    Gen2 = Embedded_GainInput(impute2, embed_cols, impute_cols2, ceiling, seed=seed )
    Generator2_Input=Gen2[0]
    Embedded2_Imputations=Gen2[1]
    
    shipment_impute = np.array(shipment_gain(Generator1_Input)[1])[:,Embedded1_Imputations[0]]
    all_predicted_Shipmentdays = np.floor(shipment_impute*(max1[0]+1e-6))
    trans_impute = np.array(transname_gain(Generator1_Input)[1])[:,Embedded1_Imputations[1]]
    all_predicted_transname = np.round(trans_impute*max1[1] + min1[1],0)
    transdays_impute = np.array(transday_gain(Generator2_Input)[1])[:,Embedded2_Imputations[0]]
    all_predicted_Transdays= np.floor(transdays_impute*(max2[0]+1e-6) + min2)
    
    all_delivered = np.full(len(transdays_impute),-1)
    all_delivered[all_predicted_Transdays>-1] = all_predicted_Shipmentdays[all_predicted_Transdays>-1] + all_predicted_Transdays[all_predicted_Transdays>-1]
    
    # number of orders
    orders= np.shape(tree_data)[0]
    
    predict_nodes = np.zeros(orders)
    # Track the node the order ends up in
    end_nodes = np.zeros(orders)
    
    # Track whether we had knowledge
    Knowledge_is_power= np.full(orders,-1)
    for order in tqdm(range(orders)):
        predicted_Shipmentdays = (int) (all_predicted_Shipmentdays[order])
        predicted_transname = (int) (all_predicted_transname[order])
        transdays_impute = (int) (all_predicted_Transdays[order])
        predicted_Transporterdays = all_predicted_Transdays[order]
        
        # Check if our current day gives already information about lagged order data
        # True data
        promiseddays = (int) (tree_data["Promised_Days"].iloc[order])
        actual_canceldays = (int) (tree_data["Canceldays"].iloc[order])
        actual_shipdays = (int) (tree_data["ShipmentDays"].iloc[order])
        actual_transdays = (int) (tree_data["TransporterDays"].iloc[order])
        actual_returndays = (int) (tree_data["ReturnDays"].iloc[order])
        actual_casedays = (int) (tree_data["CaseDays"].iloc[order])
        
        # Store true data --> based on ordering of nodes:
            # 0: canceldays
            # 2: shipmentdays
            # 3: return
            # 4: transporterdays
            # 6: case
            # 10: return
            # 11: return
            # 13: case
            # 17: case
        daynodes= np.array([actual_canceldays, actual_shipdays, 
                            actual_returndays,actual_transdays,
                            actual_casedays,actual_returndays,
                            actual_returndays,actual_casedays,
                            actual_casedays])
        
        # array used for checking if we can use actual data
        predict_days=np.full(len(daynodes),-4)
        
        # Array of nodes which we could skip if we would have the information
        skip_node_indices = np.array([0,2,3,4,6,10,11,13,17])
        skip_nodes=np.full(len(daynodes),-1)
        
        # Check which nodes could be skipped
        for skip_node in range(len(skip_nodes)):
             if(currentday>daynodes[skip_node] and daynodes[skip_node]>=0 ):
                 skip_nodes[skip_node]=1
             
        # Mark the values which we can use. 
        predict_days[np.where(skip_nodes==1)]=daynodes[np.where(skip_nodes==1)]
        
        # Set predicted transportername to None 
        # and predicted transporter element to value predicted by first GAIN
        predicted_transportername= None
        transporter_element = 1- predicted_transname
        
        # Extract actual values of shipmentdays and transporterdays (if known)
        other_predicted_Shipmentdays= predict_days[1]
        other_predicted_Transporterdays = predict_days[3]
        
        # If we can use actual shipment days, change predictions
        if(other_predicted_Shipmentdays>=0):
            predicted_Shipmentdays =  other_predicted_Shipmentdays
            transporter_element = np.sum(tree_data["transporterName"].iloc[order]==known_transporters)
        
        # If we can use actual transporter days, change predictions        
        if(other_predicted_Transporterdays>=-1):
            predicted_Transporterdays = other_predicted_Transporterdays
            skip_nodes[0]=1
        
        # Only use nodes we are able to skip
        skip_node_indices = skip_node_indices[skip_nodes>0]            
        
        # Update if we used actual data
        if(len(skip_node_indices)>0):
            if(np.sum(skip_node_indices==2)==1):
                Knowledge_is_power[order]=2
            if(np.sum(skip_node_indices==4)==1):
                Knowledge_is_power[order]=4
        
        # Nodes to keep progression of tree    
        inNode=0
        outNode=0
        
        # Fill GAIN_Tree with predicted values
        GAIN_Tree[2] = [transporter_element, predicted_Shipmentdays ]
        GAIN_Tree[4] = [predicted_Transporterdays,predicted_Shipmentdays]
        
        # Loop through tree to predict matching
        while(outNode>=0):
           inNode=outNode
           element=0
           # For these nodes, we predict the outcomes
           if(np.sum(inNode==skip_node_indices)!=1):
               # Check that we are not at an endnode
               if(len(GAIN_Tree[inNode])>1):
                   # For cancel, return and case, we use binomial distributions
                    if(inNode!=2 and inNode!=4 ):   
                        element= np.array(np.where(np.random.multinomial(1,(GAIN_Tree[inNode]))==1))[0][0]
                    # For shipment days & transporter name (type), we use first GAIN
                    if(inNode==2):
                        element= transporter_element
                    # For transporter days, we use second GAIN
                    if(inNode==4):
                        element=1
                        if(predicted_Transporterdays>=0):
                            element=0
                            predicted_Deliverydays = predicted_Shipmentdays + predicted_Transporterdays 
                            if(predicted_Deliverydays <= promiseddays ):
                                element=2
           
           # For these nodes, we know the outcomes
           if(np.sum(inNode==skip_node_indices)==1):
                 # If case or return date is known, we have unhappy match. Hence, we set element to 0
                if(inNode!=2):   
                    element= 0
                # If we know shipment days, we also know transporter name (type). 
                # Hence, set element accordingly
                if(inNode==2):          
                    element= transporter_element
                # If we know transporterdays, change elements accordingly
                if(inNode==4):
                    element=1
                    if(predicted_Transporterdays>=0):
                        element=0
                        random_day = np.random.randint(-1,1)
                        predicted_Deliverydays = predicted_Shipmentdays + predicted_Transporterdays + random_day
                        if(predicted_Deliverydays <= promiseddays ):
                            element=2
                    
           # Update outnode  
           outNode=IndexTree[inNode][element]
        
        predict_nodes[order]=outNode
        end_nodes[order]= inNode
    return([predict_nodes, end_nodes, Knowledge_is_power])
    

def TreeStats(tree_data, predict_nodes, end_nodes, 
              Knowledge_is_power, beta=1, talk=True):
    obs= len(predict_nodes)
    accuracy = np.sum(tree_data["MatchC0des"]==predict_nodes)/obs
    compare_pred = np.transpose(np.vstack((tree_data["MatchC0des"],predict_nodes, Knowledge_is_power)))
    
    
    unhappy_elements= np.where(compare_pred[:,0]==-1)
    unknown_elements= np.where(compare_pred[:,0]==-2)
    happy_elements= np.where(compare_pred[:,0]==-3)
    
    recall_unhappy = np.sum(compare_pred[unhappy_elements,0]==compare_pred[unhappy_elements,1])/len(np.array(unhappy_elements)[0])
    recall_unknown = np.sum(compare_pred[unknown_elements,0]==compare_pred[unknown_elements,1])/len(np.array(unknown_elements)[0])
    recall_happy = np.sum(compare_pred[happy_elements,0]==compare_pred[happy_elements,1])/len(np.array(happy_elements)[0])
    
    unhappy_elements= np.where(compare_pred[:,1]==-1)
    unknown_elements= np.where(compare_pred[:,1]==-2)
    happy_elements= np.where(compare_pred[:,1]==-3)
    
    precision_unhappy = np.sum(compare_pred[unhappy_elements,0]==compare_pred[unhappy_elements,1])/len(np.array(unhappy_elements)[0])
    precision_unknown = np.sum(compare_pred[unknown_elements,0]==compare_pred[unknown_elements,1])/len(np.array(unknown_elements)[0])
    precision_happy = np.sum(compare_pred[happy_elements,0]==compare_pred[happy_elements,1])/len(np.array(happy_elements)[0])
    
    F_happy = (1+beta**2)*(precision_happy*recall_happy)/(beta**2*precision_happy + recall_happy)
    F_unhappy = (1+beta**2)*(precision_unhappy*recall_unhappy)/(beta**2*precision_unhappy + recall_unhappy)
    F_unknown = (1+beta**2)*(precision_unknown*recall_unknown)/(beta**2*precision_unknown + recall_unknown)
    
    if (talk==True):
        print("Tree distribution over matches:")
        print("Happy:",(np.unique(predict_nodes,return_counts=True)[1]/obs)[0])
        print("Unknown:",(np.unique(predict_nodes,return_counts=True)[1]/obs)[1])      
        print("Unhappy:",(np.unique(predict_nodes,return_counts=True)[1]/obs)[2])
        print(" ")
        print("Overall accuracy: ", accuracy)
        print("Precision happy: ", precision_happy)
        print("Precision unknown: ", precision_unknown)
        print("Precision unhappy: ", precision_unhappy)
        print("Recall happy: ", recall_happy)
        print("Recall unknown: ", recall_unknown)
        print("Recall unhappy: ", recall_unhappy)
        print("F-score happy: ", F_happy)
        print("F-score uknown: ", F_unknown)
        print("F-score unhappy: ", F_unhappy)
    return(accuracy, F_happy, F_unknown, F_unhappy)


# Store results for multiple days
def dayResults (GAIN_type, last_day, model="Advanced"):
    days = np.arange(0,last_day)
    dayslen= len(days)
    accuracies= np.zeros(dayslen)
    F_happy = np.zeros(dayslen)
    F_unhappy = np.zeros(dayslen)
    F_unknown = np.zeros(dayslen)
    for day in tqdm(range(len(days))):
        if(GAIN_type!= "Baseline"):
            Model = GAIN_Model(tree_data, GAIN_type=GAIN_type, model=model, currentday=day,seed=seed)   
        if(GAIN_type=="Baseline"):
            Model = TreeModel(tree_data, currentday=day,seed=seed)
        Stats=TreeStats(tree_data, Model[0], Model[1], Model[2],
                        beta=1, talk=False)
        accuracies[day]=Stats[0]
        F_happy[day]=Stats[1]
        F_unhappy[day]=Stats[3]
        F_unknown[day]=Stats[2]

    return(accuracies, F_happy, F_unhappy, F_unknown)   

print("Select number of days after the order date to compute results: ")
currentday= int(input()) 
print(".")
print("You have selected to look", currentday, "days after the order date.")
seed=5071994
print("Let us find the results")
print("Results using original neural network architecture (main text) ")
# Results
print("Find results: ")
print("Baseline: ")
Baseline= TreeModel(tree_data, currentday=currentday,seed=seed)
print("GAIN: ")
GAIN = GAIN_Model(tree_data, GAIN_type="GAIN", model="BatchNorm", currentday=currentday,seed=seed)
print("WGAIN: ")
WGAIN = GAIN_Model(tree_data, GAIN_type="WGAIN", model="BatchNorm", currentday=currentday,seed=seed)
print("WGAINGP: ")
WGAINGP = GAIN_Model(tree_data, GAIN_type="WGAINGP", model="BatchNorm", currentday=currentday,seed=seed)

# # Show resuts
print(" ")
print("Baseline results: ")    
TreeStats(tree_data, Baseline[0], Baseline[1], Baseline[2]) 
print(" ")
print("GAIN results: ")    
TreeStats(tree_data, GAIN[0], GAIN[1], GAIN[2])    
print(" ")
print("WGAIN results: ")    
TreeStats(tree_data, WGAIN[0], WGAIN[1], WGAIN[2])    
print(" ")
print("WGAINGP results: ")    
TreeStats(tree_data, WGAINGP[0], WGAINGP[1], WGAINGP[2])  

print(" ")
print("Results using alternative neural network architecture (appendix) ")
# Results
print("Find results: ")
print("GAIN: ")
GAIN = GAIN_Model(tree_data, GAIN_type="GAIN", model="Advanced", currentday=currentday,seed=seed)
print("WGAIN: ")
WGAIN = GAIN_Model(tree_data, GAIN_type="WGAIN", model="Advanced", currentday=currentday,seed=seed)
print("WGAINGP: ")
WGAINGP = GAIN_Model(tree_data, GAIN_type="WGAINGP", model="Advanced", currentday=currentday,seed=seed)

# # Show resuts
print(" ")
print("Baseline results: ")    
TreeStats(tree_data, Baseline[0], Baseline[1], Baseline[2]) 
print(" ")
print("GAIN results: ")    
TreeStats(tree_data, GAIN[0], GAIN[1], GAIN[2])    
print(" ")
print("WGAIN results: ")    
TreeStats(tree_data, WGAIN[0], WGAIN[1], WGAIN[2])    
print(" ")
print("WGAINGP results: ")    
TreeStats(tree_data, WGAINGP[0], WGAINGP[1], WGAINGP[2])    
# Plot results
print(" ")
print("Do you want to plot results: ")
print("1=yes")
print("0=no")
#%%
answer=int(input()) 
if(answer==1):
    print("The answer is yes. Plotting starts now.")
    seed=5071994
    # Obtain results for multiple days
    last_day=11
    Baselinedays = dayResults("Baseline", last_day)
    GAINdays= dayResults("GAIN", last_day)
    WGAINdays= dayResults("WGAIN", last_day)
    WGAINGPdays= dayResults("WGAINGP", last_day)    
    
    
    days= np.arange(0,last_day)
    
    # Accuracies
    Baseline_Acc=Baselinedays[0]
    GAIN_Acc= GAINdays[0]
    WGAIN_Acc= WGAINdays[0]
    WGAINGP_Acc= WGAINGPdays[0]
    
    
    plt.figure(0)
    plt.plot(days, Baseline_Acc, label="Baseline", color="red")
    plt.plot(days, GAIN_Acc, label="GAIN", color="blue")
    plt.plot(days, WGAIN_Acc, label="WGAIN", color="orange")
    plt.plot(days, WGAINGP_Acc, label="WGAINGP", color="olive")
    plt.title('Accuracies after number of days after order date')
    plt.xlabel('Days')
    plt.ylabel('Accuracy')
    plt.legend()
    
    
    # F_score GAIN
    GAIN_happy = GAINdays[1]
    GAIN_unhappy = GAINdays[2]
    GAIN_unknown = GAINdays[3]
    plt.figure(1)
    plt.plot(days, GAIN_happy, label="Happy", color='green')
    plt.plot(days, GAIN_unhappy, label="Unhappy", color="red")
    plt.plot(days, GAIN_unknown, label="Unknown", color="gray")
    plt.title('F-score GAIN after number of days after order date')
    plt.xlabel('Days')
    plt.ylabel('F-score')
    plt.legend()
    
    
    # F_score WGAIN
    WGAIN_happy = WGAINdays[1]
    WGAIN_unhappy = WGAINdays[2]
    WGAIN_unknown = WGAINdays[3]
    plt.figure(2)
    plt.plot(days, WGAIN_happy, label="Happy", color='green')
    plt.plot(days, WGAIN_unhappy, label="Unhappy", color="red")
    plt.plot(days, WGAIN_unknown, label="Unknown", color="gray")
    plt.title('F-score WGAIN after number of days after order date')
    plt.xlabel('Days')
    plt.ylabel('F-score')
    plt.legend()
    
    
    # F_score WGAINGP
    WGAINGP_happy = WGAINGPdays[1]
    WGAINGP_unhappy = WGAINGPdays[2]
    WGAINGP_unknown = WGAINGPdays[3]
    plt.figure(3)
    plt.plot(days, WGAINGP_happy, label="Happy", color='green')
    plt.plot(days, WGAINGP_unhappy, label="Unhappy", color="red")
    plt.plot(days, WGAINGP_unknown, label="Unknown", color="gray")
    plt.title('F-score WGAIN-GP after number of days after order date')
    plt.xlabel('Days')
    plt.ylabel('F-score')
    plt.legend()

if(answer==0):
    print("The answer is no")
    print("That is a pity, no picasso for you")



    
    
    
    
    