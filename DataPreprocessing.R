rm(list=ls())     # Clean memory
graphics.off()    # Close graphs
cat("\014")       # Clear Console

# Functions 
#############################################

# Take sample of entire dataset
sample_taker <- function(raw_Data, N=10^3, seed=123){
  set.seed(123)
  # Shuffle data and take subsample
  shuffle <- sample(nrow(raw_Data))
  subsample <- raw_Data[shuffle[1:N],]
  
  # Convert string dates to date frame
  subsample$orderDate <- as.Date(subsample$orderDate) 
  subsample$promisedDeliveryDate <- as.Date(subsample$promisedDeliveryDate)
  subsample$registrationDateSeller <- as.Date(subsample$registrationDateSeller)

  # Convert shipment date
  known_ship = subsample[which(subsample$shipmentDate!=""),]
  known_ship$shipmentDate = as.Date(known_ship$shipmentDate) 
  unknown_ship = subsample[which(subsample$shipmentDate==""),]
  unknown_ship$shipmentDate = unknown_ship$orderDate - 1
  subsample=rbind(unknown_ship, known_ship)
  
  # Convert delivery date
  known_deliv = subsample[which(subsample$datetTimeFirstDeliveryMoment!=""),]
  known_deliv$datetTimeFirstDeliveryMoment = as.Date(known_deliv$datetTimeFirstDeliveryMoment) 
  unknown_deliv = subsample[which(subsample$datetTimeFirstDeliveryMoment==""),]
  unknown_deliv$datetTimeFirstDeliveryMoment = unknown_deliv$orderDate - 1
  subsample=rbind(known_deliv, unknown_deliv)
  
  # Convert return date
  return = subsample[which(subsample$returnDateTime!=""),]
  return$returnDateTime = as.Date(return$returnDateTime) 
  no_return = subsample[which(subsample$returnDateTime==""),]
  no_return$returnDateTime = no_return$orderDate - 1
  subsample=rbind(no_return, return)
  
  # Covert cancellation date, such that the empty-values are set to 1 day before order date
  # For no-cancels, we need to add a fake cancellation date and NO_cancelreason
  no_cancels=  subsample[which(subsample$cancellationDate== ""),]
  no_cancels$cancellationDate= no_cancels$orderDate - 1
  no_cancels$cancellationReasonCode = "NO_CANCEL"
  # For cancels, we need to convert string to date and more. 
  cancels=  subsample[which(subsample$cancellationDate!= ""),]
  # For cancellations, we need to add a fake dates and fake names for transporter 
  cancels$cancellationDate = as.Date(cancels$cancellationDate)
  # We take 2 days before order date as fake date given cancellation
  cancels$transporterCode = "NONE"
  cancels$transporterName = "NONE"
  cancels$datetTimeFirstDeliveryMoment = cancels$orderDate -2
  cancels$returnDateTime= cancels$orderDate-2
  subsample= rbind(cancels, no_cancels)
  
  # Convert case date
  case = subsample[which(subsample$startDateCase!=""),]
  case$startDateCase = as.Date(case$startDateCase) 
  no_case = subsample[which(subsample$startDateCase==""),]
  no_case$startDateCase = no_case$orderDate - 1
  subsample=rbind(no_case, case)
  
  # Add Months, weeks and weekdays to data (holidays are also possible)
  Order_month <- as.factor(month.name[as.numeric(format(subsample$orderDate, "%m"))])
  Order_week <- as.factor(strftime(subsample$orderDate, format = "%V"))
  Order_weekday <- as.factor(weekdays(subsample$orderDate))
  subsample <- cbind(subsample, Order_month, Order_week, Order_weekday)
  
  # Convert price and quantity to numeric
  subsample$totalPrice <- as.numeric(subsample$totalPrice)
  subsample$quantityOrdered <- as.numeric(subsample$quantityOrdered) 
  
  # Add price per product to data
  productPrice <- subsample$totalPrice/subsample$quantityOrdered
  subsample <- cbind(subsample, productPrice)
  
  # Add variable that takes the number of days after which an order gets cancelled 
  Promised_Days= subsample$promisedDeliveryDate -subsample$orderDate
  Canceldays = (subsample$cancellationDate - subsample$orderDate)
  ShipmentDays = subsample$shipmentDate - subsample$orderDate
  DeliveryDays= subsample$datetTimeFirstDeliveryMoment - subsample$orderDate
  ReturnDays = subsample$returnDateTime - subsample$orderDate
  CaseDays = subsample$startDateCase - subsample$orderDate
  SellerAge = subsample$orderDate - subsample$registrationDateSeller 
  subsample= cbind(subsample, Promised_Days, Canceldays, ShipmentDays, DeliveryDays, ReturnDays, CaseDays, SellerAge)
  
  # Convert delivery string into date type. 
  
  
  #########################################################
  # Create factors for categorical variables
  #########################################################
  subsample$countryCode <-as.factor(subsample$countryCode)
  subsample$transporterCode <- as.factor(subsample$transporterCode)
  subsample$transporterName <- as.factor(subsample$transporterName)
  subsample$transporterNameOther <- as.factor(subsample$transporterNameOther)
  subsample$fulfilmentType <- as.factor(subsample$fulfilmentType)
  subsample$returnCode <- as.factor(subsample$returnCode)
  subsample$cancellationReasonCode <- as.factor(subsample$cancellationReasonCode)
  subsample$productTitle <- as.factor(subsample$productTitle)
  subsample$brickName <- as.factor(subsample$brickName)
  subsample$productGroup <- as.factor(subsample$productGroup)
  subsample$productSubGroup <- as.factor(subsample$productSubGroup)
  subsample$productSubSubGroup <- as.factor(subsample$productSubSubGroup)
  subsample$countryOriginSeller <- as.factor(subsample$countryOriginSeller)
  subsample$currentCountryAvailabilitySeller <- as.factor(subsample$currentCountryAvailabilitySeller)
  subsample$calculationDefinitive <- as.factor(subsample$calculationDefinitive)
  subsample$noCancellation <- as.factor(subsample$noCancellation)
  subsample$onTimeDelivery[subsample$onTimeDelivery == ""]= "unknown"
  subsample$onTimeDelivery <- as.factor(subsample$onTimeDelivery)
  subsample$noCase <- as.factor(subsample$noCase)
  subsample$hasOneCase <- as.factor(subsample$hasOneCase)
  subsample$hasMoreCases <- as.factor(subsample$hasMoreCases)
  subsample$noReturn <- as.factor(subsample$noReturn)
  subsample$generalMatchClassification <- as.factor(subsample$generalMatchClassification)
  subsample$detailedMatchClassification <- as.factor(subsample$detailedMatchClassification)
  #########################################################
  
  # Add 0 to NA in count data
  #########################################################
  subsample$cntDistinctCaseIds[is.na(subsample$cntDistinctCaseIds)] = 0
  subsample$quanityReturned[is.na(subsample$quanityReturned)] = 0
  
  #########################################################
  
  return(subsample)
}

# Here begins data
# Set path from where the original Bol.com data is extracted (both 2019 and 2020)
setwd("C:/Users/Lars Hurkmans/Downloads/Master vakken/Case study/Data")
# Specify output path for storing data
# Set path (until map /GAIN_CODE/Data/)
out_path <- "C:/Users/Lars Hurkmans/Downloads/GAIN_CODE/Data/"


# Dataset of 2019
raw_Data2019 <- read.csv("data_2019.csv", sep=",")

raw_Data2020 <- read.csv("data_2020.csv", sep=",")

N<- 10^5*1.9
# Pick random samples for both years
sample2019 <- sample_taker(raw_Data2019, N=N, seed=123) 
sample2020 <- sample_taker(raw_Data2020, N=N, seed=123)

# Include year variable
year <- rep("2019", N)
sample2019 <- cbind(sample2019, year)

year <- rep("2020", N)
sample2020 <- cbind(sample2020, year)

# Stack samples of both years together to create total sample. Used for histograms
sample <- rbind(sample2019, sample2020)

fileName <- file.path(out_path , "CombinedRaw.csv")
ret <- write.csv(x=sample, file=fileName)
