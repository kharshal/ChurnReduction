# clear previous data
rm(list = ls())
# set active work directory 
setwd("C:/Users/Harshal/Desktop/Edwisor/Project/Project_Datafiles")

# Load Libraries
x <- c("ggplot2","gridExtra","DMwR","corrgram","DataCombine","C50","caret","randomForest","e1071","class")
lapply(x, require, character.only = TRUE)
rm(x)

##########################################Data Preprocessing###########################################

# load Churn data
churn_train <- read.csv("Train_data.csv", header = TRUE)
churn_test <- read.csv("Test_data.csv", header = TRUE)
totalData <- rbind(churn_train, churn_test)
View(totalDataData)

# Convert the numeric variables into categorical variables
totalData$area.code <- as.factor(totalData$area.code)
totalData$number.customer.service.calls <- as.factor(totalData$number.customer.service.calls)
  
for(i in 1:ncol(totalData)){
  if(class(totalData[,i]) == 'factor'){
    totalData[,i] = factor(totalData[,i], labels=(1:length(levels(factor(totalData[,i])))))
  }
}

#########################################Missing Value Analysis##########################################

missingValueAnalysis <- function(data){
  print(sum(is.na(data))) ## No further processing required as no NA values present
}

############################################Outlier Analysis#############################################
# BoxPlots 
numeric_index = sapply(totalData,is.numeric) #selecting only numeric
numeric_data = totalData[,numeric_index]
cnames = colnames(numeric_data)

# Replace all outliers with NA and impute
for(i in cnames){
  val = totalData[,i][totalData[,i] %in% boxplot.stats(totalData[,i])$out]
  totalData[,i][totalData[,i] %in% val] = NA
}
totalData = knnImputation(totalData, k = 7)

##################################Feature Selection################################################
# Correlation Plot 
numeric_index = sapply(totalData,is.numeric)
corrgram(totalData[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

# Chi-squared Test of Independence
factor_index = sapply(totalData,is.factor)
factor_data = totalData[,factor_index]

for (i in 1:(ncol(factor_data)-1)) {
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i]),simulate.p.value = TRUE))
}

# Dimension Reduction
totalData = subset(totalData, select = -c(total.day.charge,total.eve.charge,total.night.charge,total.intl.charge,
                                 area.code,phone.number))

##################################Feature Scaling################################################
#Normalisation
cnames = colnames(totalData[,sapply(totalData,is.numeric)])
for(i in cnames){
  totalData[,i] = (totalData[,i] - min(totalData[,i])) / (max(totalData[,i] - min(totalData[,i])))
}

#######################################Training Data Model########################################

DataSplit <- split(totalData, findInterval(1:nrow(totalData), 3334))
trainData <- data.frame(DataSplit['0'])
names(trainData) <- substring(names(trainData), 4)

testData <- data.frame(DataSplit['1'])
names(testData) <- substring(names(testData), 4)

missingValueAnalysis(testData)
missingValueAnalysis(trainData)

validationMatrix <- function(conf_Matrix){
  TN = conf_Matrix[1,1]
  FN = conf_Matrix[2,1]
  FP = conf_Matrix[1,2]
  TP = conf_Matrix[2,2]
  
  # Accuracy
  Accuracy = ((TN + TP) * 100) / (TN + TP + FN + FP)
  
  # False Negative Rate
  FNR = (FN / (FN + TP)) * 100
  return(c(Accuracy, FNR))
}

###################################Model Development#######################################
# Clean the environment
rmExcept(c("testData","trainData","validationMatrix"))

###############################Decision tree for classification###################################
#Develop Model on training data
DT_model = C5.0(Churn ~., trainData, trials = 100, rules = TRUE)
summary(DT_model)
#Lets predict for test cases
DT_Predict = predict(DT_model, testData[,-15], type = "class")

#Evaluate the performance of classification model
ConfMatrix_DT = table(testData$Churn, DT_Predict)

DT_valid_Matrix <- validationMatrix(ConfMatrix_DT)
print(paste0("Decision Tree Accuracy: ", DT_valid_Matrix[1]))
print(paste0("Decision Tree False Negative rate: ", DT_valid_Matrix[2]))

rmExcept(c("testData","trainData","validationMatrix","DT_valid_Matrix"))

##################################Random Forest######################################
#Random Forest
RF_model = randomForest(Churn ~ ., trainData, importance = TRUE, ntree = 500)
summary(RF_model)
#Predict test data using random forest model
RF_Predict = predict(RF_model, testData[,-15])

#Evaluate the performance of classification model
ConfMatrix_RF = table(testData$Churn, RF_Predict)

RF_valid_Matrix <- validationMatrix(ConfMatrix_RF)
print(paste0("Random Forest Accuracy: ", RF_valid_Matrix[1]))
print(paste0("Random Forest False Negative rate: ", RF_valid_Matrix[2]))

rmExcept(c("testData","trainData","validationMatrix","DT_valid_Matrix","RF_valid_Matrix"))

##################################Logisitic Regression######################################
#Logistic Regression
logit_model = glm(Churn ~ ., data = trainData, family = "binomial")
summary(logit_model)
#predict using logistic regression
logit_Predict = predict(logit_model, newdata = testData, type = "response")

#convert probabilities
logit_Predict = ifelse(logit_Predict > 0.5, 1, 0)

#Evaluate the performance of classification model
ConfMatrix_logit = table(testData$Churn, logit_Predict)

logit_valid_Matrix <- validationMatrix(ConfMatrix_logit)
print(paste0("Logistic Regression Accuracy: ", logit_valid_Matrix[1]))
print(paste0("Logistic Regression False Negative rate: ", logit_valid_Matrix[2]))

rmExcept(c("testData","trainData","validationMatrix","DT_valid_Matrix","RF_valid_Matrix",
           "logit_valid_Matrix"))

######################################KNN##########################################
#Predict test data
KNN_Predict = knn(trainData[, 1:14], testData[, 1:14], trainData$Churn, k = 7)

#Confusion matrix
ConfMatrix_knn = table(KNN_Predict, testData$Churn)

knn_valid_Matrix <- validationMatrix(ConfMatrix_knn)
print(paste0("KNN Accuracy: ", knn_valid_Matrix[1]))
print(paste0("KNN False Negative rate: ", knn_valid_Matrix[2]))

rmExcept(c("testData","trainData","validationMatrix","DT_valid_Matrix","RF_valid_Matrix",
           "logit_valid_Matrix","knn_valid_Matrix"))

##################################Naive Bayes######################################
#Develop model
NB_model = naiveBayes(Churn ~ ., data = trainData)
summary(NB_model)
#predict on test cases #raw
NB_Predict = predict(NB_model, testData[,1:14], type = 'class')

#Look at confusion matrix
ConfMatrix_NB = table(observed = testData[,15], predicted = NB_Predict)

NB_valid_Matrix <- validationMatrix(ConfMatrix_NB)
print(paste0("Naive Bayes Accuracy: ", NB_valid_Matrix[1]))
print(paste0("Naive Bayes False Negative rate: ", NB_valid_Matrix[2]))

rmExcept(c("testData","trainData","validationMatrix","DT_valid_Matrix","RF_valid_Matrix",
           "logit_valid_Matrix","knn_valid_Matrix","NB_valid_Matrix"))
