#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:41:07 2021

@author: DaanS
"""

#%% Clear workspace & console

try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass
#%% 
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm

#%% 
# Set path (Until map GAIN_CODE)
path = "C:/Users/Lars Hurkmans/Downloads/GAIN_CODE"
os.chdir(path+ "/Data/")

## Combined dataset with variables included
data_full = pd.read_csv('CombinedRaw.csv', header = 0, low_memory=False)
#%% Sample om mee te werken
data_sample = data_full.sample(n=250000, random_state = 2021)
#data_sample10 = data_full.sample(n=10, random_state = 2021)
    
#%% Voor de fun
# hoe een bepaald specifiek product vinden: data.loc[data['productSubSubGroup'] == 'Ankers']

#%% Drop irrelevant columns
# Irrelevant columns: transporterCode, transporterNameOther, productId, productTitle, brickName, productGroup, productSubGroup, productSubSubGroup, 
# Fase I

def removeIrrelevant(data_input):
    # drop transporterCode
    data_input = data_input.drop(['transporterCode'], axis = 1)
    # drop transporterNameOther
    data_input = data_input.drop(['transporterNameOther'], axis = 1)
    # productId
    data_input = data_input.drop(['productId'], axis = 1)
    # productTitle
    data_input = data_input.drop(['productTitle'], axis = 1)
    # brickName
    data_input = data_input.drop(['brickName'], axis = 1)
    # productGroup
    data_input = data_input.drop(['chunkName'], axis = 1)
    # productSubGroup
    data_input = data_input.drop(['productSubGroup'], axis = 1)
    # productSubSubGroup
    data_input = data_input.drop(['productSubSubGroup'], axis = 1)
    # productPrice
    data_input = data_input.drop(['productPrice'], axis =1)
    
    ### drop alle boolean columns
    
    # drop noCase
    data_input = data_input.drop(['noCase'], axis = 1)
    # drop hasOneCases
    data_input = data_input.drop(['hasOneCase'], axis = 1)
    # drop hasMoreCase
    data_input = data_input.drop(['hasMoreCases'], axis = 1)
    # noReturn
    data_input = data_input.drop(['noReturn'], axis = 1)
    # returnCode
    data_input = data_input.drop(['returnCode'], axis = 1)
    # cntDistinctCaseIds
    data_input = data_input.drop(['cntDistinctCaseIds'], axis = 1)
    
    ### drop alle date time variables 
    
    # drop cancelleationDate
    data_input = data_input.drop(['cancellationDate'], axis = 1)
    # drop promisedDeliveryDate
    data_input = data_input.drop(['promisedDeliveryDate'], axis = 1)
    # drop shipmentDate
    data_input = data_input.drop(['shipmentDate'], axis = 1)
    # drop dateTimeFirstDeleverymoment
    data_input = data_input.drop(['datetTimeFirstDeliveryMoment'], axis = 1)
    # drop startDateCase
    data_input = data_input.drop(['startDateCase'], axis = 1)
    # returnDateTime
    data_input = data_input.drop(['returnDateTime'], axis = 1)
    # registrationDateSeller
    data_input = data_input.drop(['registrationDateSeller'], axis = 1)
   
    return(data_input)
#%% Datetime toevoegen aan het dataframe

# order date nog even houden. De rest van de mag weg. 
def timeToString(data_input):
    data_input['onTimeDelivery'] = data_input['onTimeDelivery'].astype(str)
    
    return(data_input)
#%% Add een makkelijkere manier om de seller ID te representeren

# Create new IDs for the sellers, such that they are easier to read

from collections import defaultdict

def IDfraude(data_input):
    # the original sellerID's
    original_sellerID = data_input['sellerId'].values
    ## magic
    temp = defaultdict(lambda: len(temp))
    new_ID = [temp[ele] for ele in tqdm(original_sellerID)]
    
    data_input.insert(3, 'ID', new_ID, True)
    
    return(data_input)
#%% Remove specific data per column. 

def onlyPositivity(data_input):
    # Promised days -- alles wat negatief is eruit, en alles > 30 eruit
    data_input = data_input.loc[(data_input['Promised_Days'] >= 0) & (data_input['Promised_Days'] <= 30)]
    # Canceldays -- alles na 10 dagen weg. 
    data_input = data_input.loc[(data_input['Canceldays'] <= 10)]
    # Shipment days > 30 eruit. -1 betekent dat unknown is. -2: pakket is al gecanceld dus geen shipment
    data_input = data_input.loc[(data_input['ShipmentDays'] <= 30)]
    # DeliveryDays. -1 delivery is onbekend.  -2 cacelled dus geen shipment. Na 30 dagen weg.
    data_input = data_input.loc[(data_input['DeliveryDays'] >= -2) & (data_input['DeliveryDays'] <= 30)]
    # ReturnDays -1: geen return. en -2 zelfde als bij delivery. Alles boven de 30 eruit. 
    data_input = data_input.loc[(data_input['ReturnDays'] >= -2) & (data_input['ReturnDays'] <= 30)]
    # Casedays : -1: geen return.  En boven 30 eruit. 
    data_input = data_input.loc[(data_input['CaseDays'] >= -1) & (data_input['CaseDays'] <= 30)]
    # boven de 20 quantity halen we eruit (of een bepaald percentage)
    #data_input = data_input.loc[(data_input[''])]
    # Prijs; quantile pakken waarin we werken op basis van de log(price) kijken wat relevant is
    # Voor nu: alles tussen 5 en 2000 euro wordt meegenomen. (reden; elektronica is duur)
    lowerbound = 5
    upperbound = 2000
    data_input = data_input.loc[(data_input['totalPrice'] >= lowerbound) & (data_input['totalPrice'] <= upperbound)]
    # Alle Promised_Days < 8 eruit:  want we nemen aan dat alles daarboven 'unhappy' is en daarom niet noodzakelijk in de GAIN. Alles erboven wordt toch unhappy
    data_input = data_input.loc[(data_input['Promised_Days'] < 8)]

    return(data_input)


#%% Delivery voor bestelling? Je kan niet eerder leveren dan dat je het besteld hebt. Dat moet eruit. 
## Check script van Lars en voeg alles tot aan regel 100. 

#%% Create TransporterDays variable to use for prediction of deliverydays. 

def komJeNog(data_input):
    # Create variable transporter days, which will be used later to predict the deliverydays
    TransporterDays= np.full(np.shape(data_input)[0],100)
    # For all observations with Delivery_Days >-1, take difference between delivery days and shipment days
    subtract_obs = np.where(np.array(data_input['DeliveryDays'] >-1))[0]
    subtract_data = data_input[data_input['DeliveryDays'] >-1]
    TransporterDays[subtract_obs] = subtract_data['DeliveryDays'] - subtract_data['ShipmentDays']
    # Check that actual transporter days cannot be negative. In case we see them, label them as -3,
    # such that they can later be removed
    invaliddays = np.where(TransporterDays<0)
    TransporterDays[invaliddays]=-3
    # For all observations with delivery days -2, also include this value for transporter days
    canceldels= np.where(np.array(data_input['DeliveryDays'] ==-2))[0]
    TransporterDays[canceldels] = -2
    # Set all other TransporterDays to -1
    TransporterDays[TransporterDays==100]= -1
    
    # include transporter days to data
    data_input['TransporterDays'] = TransporterDays
    
    # Remove all observations with TransporterDays equals -3
    data_input= data_input.drop(data_input[data_input['TransporterDays'] == -3].index)
    
    return(data_input)
    

#%% String to integer 

def stringToInt(data_input):
    
    data_input['ShipmentDays'].replace({1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}, inplace=True)
    data_input['ShipmentDays'].replace({8:8}, inplace=True)
    ## Everything higher than 8 will get int 8. 
    data_input['ShipmentDays'].replace({9:8,10:8, 11:8,
                                      12:8, 13:8, 
                                      14:8, 15:8,
                                      16:8, 17:8,
                                      18:8, 19:8,
                                      20:8, 21:8,
                                      22:8, 23:8,
                                      24:8, 25:8,
                                      26:8, 27:8,
                                      28:8, 29:8,
                                      30:8}, inplace=True)
    
    # Convert transporter days such that the distribution is easier to match
    data_input['TransporterDays'].replace({1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}, inplace=True)
    data_input['TransporterDays'].replace({8:8}, inplace=True)
    ## Everything higher than 8 will get number 8. 
    data_input['TransporterDays'].replace({9:8,10:8, 11:8,
                                      12:8, 13:8, 
                                      14:8, 15:8,
                                      16:8, 17:8,
                                      18:8, 19:8,
                                      20:8, 21:8,
                                      22:8, 23:8,
                                      24:8, 25:8,
                                      26:8, 27:8,
                                      28:8, 29:8,
                                      30:8}, inplace=True)

    return(data_input)
#%% Adding new columns data

def eenBeetjeMeer(data_input):
    # quantityOrdered wordt een dummy met 0=singleOrder, 1=multipleOrder. 
    
    # year dummy. Add dummy voor 2019 and 2020 and delete the column based on this. 
    data_input = pd.concat([data_input.drop('year', axis =1), pd.get_dummies(data_input['year'])], axis =1)
    
    # country code
    data_input = pd.concat([data_input.drop('countryCode', axis =1), pd.get_dummies(data_input['countryCode'])], axis =1)
    column_indices1 = [len(data_input.columns)-2, len(data_input.columns)-1]
    new_names1 = ['Belgium', 'Netherlands']
    old_names1 = data_input.columns[column_indices1]
    data_input = data_input.rename(columns=dict(zip(old_names1, new_names1), inplace = True))
    
    # fulfilment type 
    data_input = pd.concat([data_input.drop('fulfilmentType', axis =1), pd.get_dummies(data_input['fulfilmentType'])], axis =1)
    
    # countryavailability
    data_input = pd.concat([data_input.drop('currentCountryAvailabilitySeller', axis =1), pd.get_dummies(data_input['currentCountryAvailabilitySeller'])], axis =1)
    column_indices2 = [len(data_input.columns)-3, len(data_input.columns)-2, len(data_input.columns)-1]
    new_names2 = ['ALL_availibility', 'BE_availibility', 'NL_availibility']
    old_names2 = data_input.columns[column_indices2]
    data_input = data_input.rename(columns=dict(zip(old_names2, new_names2), inplace = True))
    
    return(data_input)


#%% Christmas maker
# output is 'christ_array'

def christmasIsComing(data_input):
    christmas = []
    christmas_2019 = datetime.date(2019,12,12)
    christmas_2020 = datetime.date(2020,12,12)
    days = 15
    for day in tqdm(range(days)):
        date = (christmas_2019 + datetime.timedelta(days=day)).isoformat()
        date2 = (christmas_2020 + datetime.timedelta(days=day)).isoformat()
        christmas.append(date)
        christmas.append(date2)
    
        christ_array = np.array(christmas)
    
    return(christ_array)
#%% Christmas adder 
def addChristmas(data_input, christ_array):    
    datums = data_input['orderDate'].values
    zeros = np.zeros(data_input.shape[0])
    matrix = np.transpose(np.vstack((datums, zeros)))
    
    for i in tqdm(range(data_input['orderDate'].shape[0])):
        for j in range(christmas.shape[0]):
            if data_input['orderDate'].iloc[i] == christmas[j]:
                matrix[i][1] = 1
    
    # extract the Christmas column and add to data_input
    christmas_dummy = matrix[:, 1]
    data_input.insert(10, 'Christmas', christmas_dummy)
    
    return(data_input)
#%% Easter maker
## output easter_array   
def easterEgg(data_input):
    easter = []
    easter_start_19 = datetime.date(2019, 4, 7)
    easter_start_20 = datetime.date(2020, 3, 30)
    days = 15
    for day in tqdm(range(days)):
        date = (easter_start_19 + datetime.timedelta(days=day)).isoformat()
        date2 = (easter_start_20 + datetime.timedelta(days=day)).isoformat()
        easter.append(date)
        easter.append(date2)
        
        easter_array = np.array(easter)
    
    return(easter_array)
#%%    
def addEaster(data_input, easter_array):
    datums = data_input['orderDate'].values
    zeros = np.zeros(data_input.shape[0])
    matrix = np.transpose(np.vstack((datums, zeros)))
    
    for i in tqdm(range(data_input['orderDate'].shape[0])):
        for j in range(easter.shape[0]):
            if data_input['orderDate'].iloc[i] == easter[j]:
                matrix[i][1] = 1
    
    # extract the Christmas column and add to data_input
    easter_dummy = matrix[:, 1]
    data_input.insert(10, 'Easter', easter_dummy)
    
    return(data_input)            
#%% Sinterklaas maker 

def prepareSinterklaas(data_input):
    sinterklaas = []
    sinterklaas_start_19 = datetime.date(2019, 11, 21)
    sinterklaar_start_20 = datetime.date(2020, 11, 21)
    days = 14
    for day in tqdm(range(days)):
        date = (sinterklaas_start_19 + datetime.timedelta(days=day)).isoformat()
        date2 = (sinterklaar_start_20 + datetime.timedelta(days=day)).isoformat()
        sinterklaas.append(date)
        sinterklaas.append(date2)
        
        sinterklaas_array = np.array(sinterklaas)
        
    return(sinterklaas_array)
#%% 
def addSinterklaas(data_input, sinterklaas_array):
    datums = data_input['orderDate'].values
    zeros = np.zeros(data_input.shape[0])
    matrix = np.transpose(np.vstack((datums, zeros)))
    
    for i in tqdm(range(data_input['orderDate'].shape[0])):
        for j in range(sinterklaas.shape[0]):
            if data_input['orderDate'].iloc[i] == sinterklaas[j]:
                matrix[i][1] = 1
    
    # extract the Christmas column and add to data_input
    sinterklaas_dummy = matrix[:, 1]
    data_input.insert(10, 'Sinterklaas', sinterklaas_dummy)
    
    return(data_input)
#%% Run alles in dit blok voor output: data.

## choose input: data_sample or data_full
data_input = data_sample
print('######')
print('Jetzt geht s los! ')
print('######')
data = removeIrrelevant(data_input)
print('Irrelevant columns are gone.')
print('######')
data = timeToString(data)
print('onTimeDelivery are strings.')
print('######')
data = onlyPositivity(data)
print('Restrictions in data are set.')
print('######')
data = komJeNog(data)
print('TransporterDays is added tot data.')
print('######')
data = stringToInt(data)
print('ShipmentDays and TransportDays are set to integers.')
print('######')
data = eenBeetjeMeer(data)
print('Dummy variables are added.')
print('######')

## Vanaf hier duurt het runnen wat langer. 

christmas = christmasIsComing(data)
print('Christmas dates are created.')
print('######')
data = addChristmas(data, christmas)
print('Christmas dates are added to data.')
print('######')
easter = easterEgg(data)
print('Easter dates are created.')
print('######')
data = addEaster(data, easter)
print('Easter dates are added to data.')
print('######')
sinterklaas = prepareSinterklaas(data)
print('Sinterklaas dates are created.')
print('######')
data = addSinterklaas(data, sinterklaas)
print('Sinterklaas dates are added to data.')
print('######')
data = IDfraude(data)
print('New Seller ID is made.')
print('#####')
print('Fin')


#%% Write dataframe to csv. 
# data.to_parquet('data.parquet.gzip', compression = 'gzip')
data.to_csv('data_sample_full.csv', encoding = 'utf-8')

        
#%% Handige Lars tips 

x = [1,2,3,4,5]
x = np.array(x)

y = [3,4,5,6,9]
y = np.array(y)
# geeft voor alle x groter dan 4 een waarde 1
y[x>4] = 1

# als je z=x doet moet je altijd z=x.copy() anders verandert de x mee. 

#%% 
##%% Output the data to parquet to keep the data types alive. 
# First install: conda install pyarrow
# First install: conda install fastparquet
# Output data to parquet in gzip format. 


## read data as: data = pd.read_parquet('data.parquet.gzip')

#%% 


#%% GAIN 1
# Embedding 1
#   sellerID
# Embedding 2
#   Chunkname
# Embedding 3 
#   order Month
# Embedding 4
#   order Week 
# NOT IMPUTE GAIN 1
#   total price, quantity ordered, country code, fulfilment type, country origin seller, country avail. seller, seller age, year, holidays (4 kolommen)
# IMPUTE 
#   shipmentdays en transporterName

#%% Van GAIN 1 naar GAIN 2 
# alle -1, -2 data van shipment days eruit.
# alle transporters met unknown eruit: Anders, Bezorgafspraak, Briefpost, BPost Briefpost, DHL Global Mail, Dynalogic, NONE, 
# Packs, Parcel.NL, TNT Express, TransMission, Transport Service Nederland, UPS.

#%% GAIN 2
# Embedding 1
#   sellerID
# Embedding 2
#   Chunkname
# Embedding 3 
#   order Month
# Embedding 4
#   order Week 
# Embedding 5
#   Transporter Name (neem als gegeven niet uit eerste GAIN, dus onafh.) (alleen datat zoals beschreven hierboven)
# NOT IMPUTE GAIN 2
#   total price, quantity ordered, country code, fulfilment type, country origin seller, country avail. seller, seller age, year, holidays (4 kolommen),
#   shipmentdays 
# IMPUTE
#   deliveryday (time for the transporter to reach destination) 'TRANSPORTDAYS' = delivery - shipment. 

