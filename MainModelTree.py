# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:35:55 2021

@author: Lars Hurkmans
"""

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
empirical_case1= np.array(pd.read_csv(empirical_path + "case1.csv")).flatten()
empirical_case2= np.array(pd.read_csv(empirical_path + "case2.csv")).flatten()
empirical_delivery= np.array(pd.read_csv(empirical_path + "delivery.csv")).flatten()
un_del=empirical_delivery[2]
on_del=empirical_delivery[1]
empirical_delivery[1]=un_del
empirical_delivery[2]=on_del

# Product type used
product_type="productGroup"

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

# Here we create the tree model: This is based on an adjacency list
# This tree will used to track where the order ends up
IndexTree = [
    [2,1],              # Node 0  --> Order --> Node 1 or 2
    [-1],               # Node 1  --> UNHAPPY: cancel --> END NODE (-1)
    [3,4,9],            # Node 2  --> Imputation delivery --> Node  3, 4, 5 (late, unknown, on time)
    [-1],               # Node 3  --> Late delivery --> END NDOE (-1)
    [5,6],              # Node 4  --> Unknown delivery: Node 5 or 6 (return or no return)
    [-1],               # Node 5  --> UNHAPPY: Unknown delivery & Return --> END NODE(-1)
    [7,8],              # Node 6  --> Unknown & no return --> Node 7 or 8 (case or no case)
    [-1],               # Node 7  --> UNHAPPY: Unknown delivery & case --> END NODE (-1)
    [-2],               # Node 8  --> UNKNOWN: Unknown transporter & no return or case --> END NODE (-2)
    [10,11],            # Node 9  --> On time delivery --> Node 10 or 11 (return or no return)
    [-1],               # Node 10 --> UNHAPPY: On time but return --> END NODE (-1)
    [12,13],            # Node 11 --> On time delivery and no return --> node 12 or 13 (case or no case)
    [-1],               # Node 12 --> UNHAPPY: Unknown delivery & return --> END Node (-1)
    [-3],               # Node 13 --> HAPPY On time delivery, no return and no case END NODE (-3)
            ]               


# Baseline tree model. Here we use multinomial sampling in all nodes.  
def TreeModel(tree_data,seed=123):
    # Set seed
    np.random.seed(seed)
    # The baseline tree
    # match prediction nodes:
    # -1 = UNHAPPY
    # -2 = UNKNOWN
    # -3 = HAPPY    
    Baseline = [
                empirical_cancel,                 # Node 0  --> Order --> Node 1 or 2
                [-1],                             # Node 1  --> UNHAPPY: cancel --> END NODE (-1)
                empirical_delivery,               # Node 2  --> Imputation delivery --> Node  3, 4, 5 (late, unknown, on time)
                [-1],                             # Node 3  --> Late delivery --> END NDOE (-1)
                empirical_return1,                # Node 4  --> Unknown delivery: Node 5 or 6 (return or no return)
                [-1],                             # Node 5  --> UNHAPPY: Unknown delivery & Return --> END NODE(-1)
                empirical_case1,                  # Node 6  --> Unknown & no return --> Node 7 or 8 (case or no case)
                [-1],                             # Node 7  --> UNHAPPY: Unknown delivery & case --> END NODE (-1)
                [-2],                             # Node 8  --> UNKNOWN: Unknown transporter & no return or case --> END NODE (-2)
                empirical_return2,                # Node 9  --> On time delivery --> Node 10 or 11 (return or no return)
                [-1],                             # Node 10 --> UNHAPPY: On time but return --> END NODE (-1)
                empirical_case2,                  # Node 11 --> On time delivery and no return --> node 12 or 13 (case or no case)
                [-1],                             # Node 12 --> UNHAPPY: Unknown delivery & return --> END Node (-1)
                [-3],                             # Node 13 --> HAPPY On time delivery, no return and no case END NODE (-3)
                    ]    
    
    # Number of orders of tree data
    obs = np.shape(tree_data)[0]
    predict_nodes = np.zeros(obs)
    # Track the node the order ends up in
    end_nodes = np.zeros(obs)
        
    # predict matching for every order in tree data
    for order in tqdm(range(obs)):
    
                      
        # Nodes to keep progression of tree:
        # InNode= current node
        # outNode= new node
        inNode=0
        outNode=0
        
        # Loop through tree to predict matching
        while(outNode>=0):
           inNode=outNode
           element=0
           # Check that we are not at end node
           if(len(Baseline[inNode])>1): 
               element= np.array(np.where(np.random.multinomial(1,(Baseline[inNode]))==1))[0][0]       
           outNode=IndexTree[inNode][element]
           predict_nodes[order]=outNode
           end_nodes[order]= inNode
    return((predict_nodes, end_nodes))



def GAIN_predict(data, generator, embed_cols, impute_cols, val_min, val_max, seed ):
    tree, impute, mask = data_Imputer(data, 0.99, impute_cols)
    
    ceiling=[10,3,3,3]
    
    Gen = Embedded_GainInput(impute, embed_cols, impute_cols, ceiling, seed=seed )
    Generator_Input=Gen[0]
    Embedded_Imputations=Gen[1]    
    impute_data = np.array(generator(Generator_Input)[1])[:,Embedded_Imputations[0]]
    predictions = np.round(impute_data*(val_max+1e-6) + val_min)
    return(predictions)

# Trees of GAIN
def GAIN_Model(tree_data, GAIN_type, currentday=0, seed=123):
    # Set seed
    np.random.seed(seed)
    # Extract the generators to impute the lagged data
    delivery_gain = tf.keras.models.load_model(generator_path + "/" +GAIN_type+"/Del", compile=False)
    return1_gain= tf.keras.models.load_model(generator_path  + "/" +GAIN_type+"/Return1", compile=False)
    return2_gain= tf.keras.models.load_model(generator_path + "/" +GAIN_type+"/Return2", compile=False)
    case1_gain= tf.keras.models.load_model(generator_path  + "/" +GAIN_type+"/Case1", compile=False)
    case2_gain= tf.keras.models.load_model(generator_path  + "/" +GAIN_type+"/Case2", compile=False)
    
    # Get gain data for tree. Needed for normalisation of parameters
    delivery_tree=np.array(tree_data.drop(dropcols,axis=1))
    return1_tree= np.array(tree_data.drop(return1drop,axis=1))
    return2_tree= np.array(tree_data.drop(returndrop,axis=1))
    case1_tree= np.array(tree_data.drop(case1drop,axis=1))
    case2_tree= np.array(tree_data.drop(casedrop,axis=1))
    
    
    # Initialise imputation variables
    predicted_delivery = None
    predicted_return1 = None
    predicted_return2 = None
    predicted_case1 = None
    predicted_case2 = None
    
    
    # GAIN model
    # match prediction nodes:
        # -1 = UNHAPPY
        # -2 = UNKNOWN
        # -3 = HAPPY
    GAIN_Tree = [
                empirical_cancel,                 # Node 0  --> Order --> Node 1 or 2
                [-1],                             # Node 1  --> UNHAPPY: cancel --> END NODE (-1)
                empirical_delivery,               # Node 2  --> Imputation delivery --> Node  3, 4, 5 (late, unknown, on time)
                [-1],                             # Node 3  --> Late delivery --> END NDOE (-1)
                empirical_return1,                # Node 4  --> Unknown delivery: Node 5 or 6 (return or no return)
                [-1],                             # Node 5  --> UNHAPPY: Unknown delivery & Return --> END NODE(-1)
                empirical_case1,                  # Node 6  --> Unknown & no return --> Node 7 or 8 (case or no case)
                [-1],                             # Node 7  --> UNHAPPY: Unknown delivery & case --> END NODE (-1)
                [-2],                             # Node 8  --> UNKNOWN: Unknown transporter & no return or case --> END NODE (-2)
                empirical_return2,                # Node 9  --> On time delivery --> Node 10 or 11 (return or no return)
                [-1],                             # Node 10 --> UNHAPPY: On time but return --> END NODE (-1)
                empirical_case2,                  # Node 11 --> On time delivery and no return --> node 12 or 13 (case or no case)
                [-1],                             # Node 12 --> UNHAPPY: Unknown delivery & return --> END Node (-1)
                [-3],                             # Node 13 --> HAPPY On time delivery, no return and no case END NODE (-3)
                    ]    
    

    
    # Predict values of leading variables
    delivery_predict= GAIN_predict(delivery_tree, delivery_gain, [6,15,16,17], [5], 0,2,seed)
    return1_predict= (GAIN_predict(return1_tree, return1_gain, [5,15,16],[14],0,1, seed) -1)*-1
    return2_predict= (GAIN_predict(return2_tree, return2_gain, [5,15,16,17],[14],0,1, seed) - 1)*-1
    case1_predict= (GAIN_predict(case1_tree, case1_gain, [5,15,16],[14],0,1, seed)-1)*-1
    case2_predict= (GAIN_predict(case2_tree, case2_gain, [5,15,16,17],[14],0,1, seed)-1)*-1
    
    # number of orders
    orders= np.shape(tree_data)[0]
    
    predict_nodes = np.zeros(orders)
    # Track the node the order ends up in
    end_nodes = np.zeros(orders)
    
    # Track whether we had knowledge
    Knowledge_is_power= np.full(orders,-1)
    for order in tqdm(range(orders)):
        predicted_delivery = (int) (delivery_predict[order])
        predicted_return1 = (int) (return1_predict[order])
        predicted_return2 = (int) (return2_predict[order])
        predicted_case1 = (int) (case1_predict[order])
        predicted_case2 = (int) (case2_predict[order])

        # Nodes to keep progression of tree    
        inNode=0
        outNode=0
        

        # Loop through tree to predict matching
        while(outNode>=0):
           inNode=outNode
           element=0
           # For these nodes, we predict the outcomes
           # Check that we are not at an endnode
           if(len(GAIN_Tree[inNode])>1):
                # For cancel, return and case, we use binomial distributions
                 if(inNode==0):   
                     element= np.array(np.where(np.random.multinomial(1,(GAIN_Tree[inNode]))==1))[0][0]
                 # For delivery type, we use imputation
                 if(inNode==2):
                     if(predicted_delivery==0):
                         element=0
                     if(predicted_delivery==1):
                         element=2
                     if(predicted_delivery==2):
                         element=1
                 # Imputation for case and return
                 if (inNode==4):
                     element= predicted_return1
                 if (inNode==6):
                     element= predicted_case1
                 if (inNode==9):
                     element=  predicted_return2
                 if (inNode==11):
                     element=  predicted_case2
                 
           # Update outnode  
           outNode=IndexTree[inNode][element]
        
        predict_nodes[order]=outNode
        end_nodes[order]= inNode
    return([predict_nodes, end_nodes, Knowledge_is_power])
    

def StatsDelivery (GAIN_type, beta=1):
    np.random.seed(129)
    if (GAIN_type!= "Baseline"):
         # Extract the generators to impute the lagged data
        delivery_gain = tf.keras.models.load_model(generator_path + "/" +GAIN_type+"/Del", compile=False)
        
        # Get gain data for tree. Needed for normalisation of parameters
        delivery_tree=np.array(tree_data.drop(dropcols,axis=1))
         # Predict values of leading variables
        delivery_predict= GAIN_predict(delivery_tree, delivery_gain, [6,15,16,17], [5], 0,2,seed)
    
    if (GAIN_type=="Baseline"):
        N= np.shape(tree_data)[0]
        delivery_predict = np.random.choice(3,N,p=(np.array(empirical_delivery)))
    obs= len(delivery_predict)
    accuracy = np.sum(tree_data["onTimeDelivery"]==delivery_predict)/obs
    compare_pred = np.transpose(np.vstack((tree_data["onTimeDelivery"],delivery_predict)))
    
    
    
    unhappy_elements= np.where(compare_pred[:,0]==0)
    unknown_elements= np.where(compare_pred[:,0]==1)
    happy_elements= np.where(compare_pred[:,0]==2)
    
    recall_unhappy = np.sum(compare_pred[unhappy_elements,0]==compare_pred[unhappy_elements,1])/len(np.array(unhappy_elements)[0])
    recall_unknown = np.sum(compare_pred[unknown_elements,0]==compare_pred[unknown_elements,1])/len(np.array(unknown_elements)[0])
    recall_happy = np.sum(compare_pred[happy_elements,0]==compare_pred[happy_elements,1])/len(np.array(happy_elements)[0])
    
    unhappy_elements= np.where(compare_pred[:,1]==-0)
    unknown_elements= np.where(compare_pred[:,1]==1)
    happy_elements= np.where(compare_pred[:,1]==2)
    
    precision_unhappy = np.sum(compare_pred[unhappy_elements,0]==compare_pred[unhappy_elements,1])/len(np.array(unhappy_elements)[0])
    precision_unknown = np.sum(compare_pred[unknown_elements,0]==compare_pred[unknown_elements,1])/len(np.array(unknown_elements)[0])
    precision_happy = np.sum(compare_pred[happy_elements,0]==compare_pred[happy_elements,1])/len(np.array(happy_elements)[0])
    
    F_happy = (1+beta**2)*(precision_happy*recall_happy)/(beta**2*precision_happy + recall_happy)
    F_unhappy = (1+beta**2)*(precision_unhappy*recall_unhappy)/(beta**2*precision_unhappy + recall_unhappy)
    F_unknown = (1+beta**2)*(precision_unknown*recall_unknown)/(beta**2*precision_unknown + recall_unknown)
    
    print("Performance Delivery Date:")
    print(" ")
    print("Overall accuracy: ", np.round(accuracy,3))
    print("Precision time: ", np.round(precision_happy,3))
    print("Precision unknown: ", np.round(precision_unknown,3))
    print("Precision late: ", np.round(precision_unhappy,3))
    print("Recall time: ", np.round(recall_happy,3))
    print("Recall unknown: ", np.round(recall_unknown,3))
    print("Recall late: ", np.round(recall_unhappy,3))
    print("F-score time: ", np.round(F_happy,3))
    print("F-score uknown: ", np.round(F_unknown,3))
    print("F-score late: ", np.round(F_unhappy,3))
    print("Macro F-score:" , np.round(F_happy*empirical_delivery[2]+F_unknown*empirical_delivery[1]+F_unhappy*empirical_delivery[0],3))
    
    
def StatsBinary (GAIN_type, beta=1):
    
    np.random.seed(129)
    def Fscore (compare, beta=beta):
        # Compute F-scores for label 0 and label 1 and take then the average
        actual= compare[:,0]
        imputed= compare[:,1]
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
        return(F1,F0)
    
    
    N= np.shape(tree_data)[0]
    cancel_predict = np.random.choice(2,N,p=(1-np.array(empirical_cancel)))
    if (GAIN_type!= "Baseline"):
        return1_gain= tf.keras.models.load_model(generator_path  + "/" +GAIN_type+"/Return1", compile=False)
        return2_gain= tf.keras.models.load_model(generator_path + "/" +GAIN_type+"/Return2", compile=False)
        case1_gain= tf.keras.models.load_model(generator_path  + "/" +GAIN_type+"/Case1", compile=False)
        case2_gain= tf.keras.models.load_model(generator_path  + "/" +GAIN_type+"/Case2", compile=False)
        
        return1_tree= np.array(tree_data.drop(return1drop,axis=1))
        return2_tree= np.array(tree_data.drop(returndrop,axis=1))
        case1_tree= np.array(tree_data.drop(case1drop,axis=1))
        case2_tree= np.array(tree_data.drop(casedrop,axis=1))
        
        return1_predict= (GAIN_predict(return1_tree, return1_gain, [5,15,16],[14],0,1, seed))
        return2_predict= (GAIN_predict(return2_tree, return2_gain, [5,15,16,17],[14],0,1, seed))
        case1_predict= (GAIN_predict(case1_tree, case1_gain, [5,15,16],[14],0,1, seed))
        case2_predict= (GAIN_predict(case2_tree, case2_gain, [5,15,16,17],[14],0,1, seed))
        
    
    if (GAIN_type=="Baseline"):
        N= np.shape(tree_data)[0]

        return1_predict = np.random.choice(2,N,p=(1-np.array(empirical_return1)))
        case1_predict = np.random.choice(2,N,p=(1-np.array(empirical_case1)))
        
        return2_predict = np.random.choice(2,N,p=(1-np.array(empirical_return2)))
        case2_predict = np.random.choice(2,N,p=(1-np.array(empirical_case2)))
    
    
    compare_pred_cancel=np.transpose(np.vstack((tree_data["noCancellation"],cancel_predict)))
    compare_pred_ret1 = np.transpose(np.vstack((tree_data["Return"],return1_predict)))
    compare_pred_ret2 = np.transpose(np.vstack((tree_data["Return"],return2_predict)))
    compare_pred_case1 = np.transpose(np.vstack((tree_data["Case"],case1_predict)))
    compare_pred_case2 = np.transpose(np.vstack((tree_data["Case"],case2_predict)))
    F_cancel=Fscore(compare_pred_cancel, beta=beta)
    F_ret1= Fscore(compare_pred_ret1, beta=beta)
    F_ret2= Fscore(compare_pred_ret2, beta=beta)
    F_case1= Fscore(compare_pred_case1, beta=beta)
    F_case2= Fscore(compare_pred_case2, beta=beta)
    
    print("Accuracy Cancellation: ", np.sum(compare_pred_cancel[:,0]==compare_pred_cancel[:,1])/N)
    print("F1 cancel (no cancel): ", np.round(F_cancel[0],3))
    print("F1 return 1 (cancel): ", np.round(F_cancel[1],3))
    print("Weighted-F1 cancel: ", np.round(empirical_cancel[0]*F_cancel[0] + empirical_cancel[1]*F_cancel[1],3))
    print("############################################")
    print("Accuracy return1: ", np.sum(compare_pred_ret1[:,0]==compare_pred_ret1[:,1])/N)
    print("F1 return 1 (no return): ", np.round(F_ret1[1],3))
    print("F1 return 1 (return): ", np.round(F_ret2[0],3))
    print("Weighted-F1 return 1: ", np.round(empirical_return1[0]*F_ret1[0] + empirical_return1[1]*F_ret1[1],3))
    print("############################################")
    print("Accuracy case 1: ", np.sum(compare_pred_case1[:,0]==compare_pred_case1[:,1])/N)
    print("F1 case 1 (no case)", np.round(F_case1[1],3))
    print("F1 case 1 (case)", np.round(F_case2[0],3))
    print("Weighted-F1 case 1: ", np.round(empirical_case1[0]*F_case1[0] + empirical_case1[1]*F_case1[1],3))
    print("############################################")
    print("Accuracy return 2: ", np.sum(compare_pred_ret2[:,0]==compare_pred_ret2[:,1])/N)
    print("F1 return 2 (no return): ", np.round(F_ret1[1],3))
    print("F1 return 2 (return): ", np.round(F_ret2[0],3))
    print("Weighted-F1 return 2: ", np.round(empirical_return1[0]*F_ret1[0] + empirical_return1[1]*F_ret1[1],3))
    print("############################################")
    print("Accuracy case 2: ", np.sum(compare_pred_case2[:,0]==compare_pred_case2[:,1])/N)
    print("F1 case 2 (no case)", np.round(F_case1[1],3))
    print("F1 case 2 (case)", np.round(F_case2[0],3))
    print("Weighted-F1 return 1: ", np.round(empirical_case1[0]*F_case1[0] + empirical_case1[1]*F_case1[1],3))


def TreeStats(tree_data, predict_nodes, end_nodes, 
              beta=1, talk=True):
    obs= len(predict_nodes)
    accuracy = np.sum(tree_data["MatchC0des"]==predict_nodes)/obs
    compare_pred = np.transpose(np.vstack((tree_data["MatchC0des"],predict_nodes)))
    
    
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
        print("Overall accuracy: ", np.round(accuracy,3))
        print("Precision happy: ", np.round(precision_happy,3))
        print("Precision unknown: ", np.round(precision_unknown,3))
        print("Precision unhappy: ", np.round(precision_unhappy,3))
        print("Recall happy: ", np.round(recall_happy,3))
        print("Recall unknown: ", np.round(recall_unknown,3))
        print("Recall unhappy: ", np.round(recall_unhappy,3))
        print("F-score happy: ", np.round(F_happy,3))
        print("F-score uknown: ", np.round(F_unknown,3))
        print("F-score unhappy: ", np.round(F_unhappy,3))
        print("Weighted F-score:" , np.round(F_happy*0.57+F_unknown*.31+F_unhappy*0.12,3))
    return(accuracy, F_happy, F_unknown, F_unhappy)


seed=5071994
print("Let us find the results")
print("Results using original neural network architecture (main text) ")
# Results
print("Find results: ")
print("Baseline: ")
Baseline= TreeModel(tree_data,seed=seed)
print("GAIN: ")
GAIN = GAIN_Model(tree_data, GAIN_type="GAIN",seed=seed)
print("WGAIN: ")
WGAIN = GAIN_Model(tree_data, GAIN_type="WGAIN",seed=seed)
print("WGAINGP: ")
WGAINGP = GAIN_Model(tree_data, GAIN_type="WGAINGP", seed=seed)

# # Show resuts

# Results binary
print("Results binaries")
print(" ")
StatsBinary("Baseline")
print(" ")
print("GAIN")
StatsBinary("GAIN")
print(" ")
print("WGAIN")
StatsBinary("WGAIN")
print(" ")
print("WGAINGP")
StatsBinary("WGAINGP")


# Results delivery
print("Results delivery date")
print(" ")
print("Baseline")
StatsDelivery("Baseline")
print(" ")
print("GAIN")
StatsDelivery("GAIN")
print(" ")
print("WGAIN")
StatsDelivery("WGAIN")
print(" ")
print("WGAINGP")
StatsDelivery("WGAINGP")

print(" ")
print("Baseline results: ")    
TreeStats(tree_data, Baseline[0], Baseline[1]) 
print(" ")
print("GAIN results: ")    
TreeStats(tree_data, GAIN[0], GAIN[1])    
print(" ")
print("WGAIN results: ")    
TreeStats(tree_data, WGAIN[0], WGAIN[1])    
print(" ")
print("WGAINGP results: ")    
TreeStats(tree_data, WGAINGP[0], WGAINGP[1])  


    
    
    