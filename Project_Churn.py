# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:20:11 2018

@author: Harshal
"""

# Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fancyimpute import KNN
from scipy.stats import chi2_contingency

# Set active Work Directory
os.chdir("C:/Users/Harshal/Desktop/Edwisor/Project/Project_Datafiles")

# Load Data
churn_train = pd.read_csv("Train_data.csv")
churn_test = pd.read_csv("Test_data.csv")

# Combine Data for Preprocessing
churn_data_frames = [churn_train, churn_test]
churn_data = pd.concat(churn_data_frames)

# Exploratory Data Analysis 
# Convert Data into required types
churn_data['area code'] = churn_data['area code'].astype(str)
churn_data['number customer service calls'] = churn_data['number customer service calls'].astype(str)
    
for i in range(0, len(churn_data.columns)):
    if(churn_data.iloc[:,i].dtypes == 'object'):
        churn_data.iloc[:,i] = pd.Categorical(churn_data.iloc[:,i])
        churn_data.iloc[:,i] = churn_data.iloc[:,i].cat.codes 
        churn_data.iloc[:,i] = churn_data.iloc[:,i].astype('object')

object_cnames = []
numeric_cnames = []
target_cnames = []
for i in range(0, len(churn_data.columns)):
    if(churn_data.iloc[:,i].dtypes == 'object' and churn_data.columns[i] != 'Churn'):
        object_cnames.append(churn_data.columns[i])
    elif(churn_data.iloc[:,i].dtypes == 'int64' or churn_data.iloc[:,i].dtypes == 'float64'):
        numeric_cnames.append(churn_data.columns[i])
    else:
        target_cnames.append(churn_data.columns[i])

# Data Preprocessing
# Missing Value Analysis
def missingValueAnalysis(data):
    print(data.isna().sum())

# Outlier Analysis
# Create BoxPlot for numeric values
numeric_data = []
figure = plt.figure(1,figsize=(10,8))
fig_1 = figure.add_subplot(111)
for i in range(0, len(numeric_cnames)):
    numeric_data.append(churn_data[numeric_cnames[i]])

# Plot BoxPlot
boxPlot = fig_1.boxplot(numeric_data)

# Impute Outliers with NA
for i in numeric_cnames:
    q75, q25 = np.nanpercentile(churn_data.loc[:,i],[75, 25])
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    churn_data.loc[churn_data[i] < min, i] = np.nan
    churn_data.loc[churn_data[i] > max, i] = np.nan
#Impute with KNN
churn_data = pd.DataFrame(KNN(k = 7).complete(churn_data), columns = churn_data.columns)
missingValueAnalysis(churn_data)

# Feature Selection
#Correlation plot
churn_corr = churn_data.corr()
fig, ax = plt.subplots(figsize=(10, 8))
ax.matshow(churn_corr)
plt.xticks(range(len(churn_data.columns)), churn_data.columns, rotation='vertical')
plt.yticks(range(len(churn_data.columns)), churn_data.columns)

# chi square test values
for i in object_cnames:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(churn_data['Churn'], churn_data[i]))
    print(p)

# Dimensionality Reduction
churn_data = churn_data.drop(['total day charge', 'total eve charge', 'total night charge', 'total intl charge', 
                                'area code', 'phone number'], axis=1)

removelist = ['total day charge', 'total eve charge', 'total night charge', 'total intl charge', 'area code', 'phone number']
object_cnames = [v for i, v in enumerate(object_cnames) if v not in removelist]
numeric_cnames = [v for i, v in enumerate(numeric_cnames) if v not in removelist]

# Feature Scaling
#Nomalisation
for i in numeric_cnames:
    churn_data[i] = (churn_data[i] - np.min(churn_data[i])) / (np.max(churn_data[i]) - np.min(churn_data[i]))

# Model Development
def validationMatrix(confMatrix):
    #let us save TP, TN, FP, FN
    TN = confMatrix.iloc[0,0]
    FN = confMatrix.iloc[1,0]
    TP = confMatrix.iloc[1,1]
    FP = confMatrix.iloc[0,1]

    #check accuracy of model
    Accuracy = ((TP+TN)*100)/(TP+TN+FP+FN)

    #False Negative rate 
    FNR = (FN*100)/(FN+TP)
    return(Accuracy,FNR)

# Decision Tree
from sklearn import tree
from sklearn.metrics import accuracy_score
DT_data = churn_data.copy()
#replace target categories with Yes or No
DT_data['Churn'] = DT_data['Churn'].replace(0, 'No')
DT_data['Churn'] = DT_data['Churn'].replace(1, 'Yes')

#Divide data into train and test
train_DT = DT_data.iloc[:3333, :]
test_DT = DT_data.iloc[3333:, :]

X_train = train_DT.values[:, 0:14]
Y_train = train_DT.values[:,14]
X_test = test_DT.values[:, 0:14]
Y_test = test_DT.values[:,14]

DT_model = tree.DecisionTreeClassifier(criterion="entropy")
DT_model = DT_model.fit(X_train, Y_train)
DT_predict = DT_model.predict(X_test)
accuracy_score(Y_test, DT_predict)
#build confusion matrix
from sklearn.metrics import confusion_matrix 
DT_CM = pd.DataFrame(confusion_matrix(Y_test, DT_predict))
DT_Accuracy, DT_FNR = validationMatrix(DT_CM)
print("Decision Tree confusion matrix\n", DT_CM)
print("\nDecision Tree Accuracy",DT_Accuracy)
print("Decision Tree False Negative rate",DT_FNR)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 500)
RF_model = RF_model.fit(X_train, Y_train)
RF_Predict = RF_model.predict(X_test)
accuracy_score(Y_test, RF_Predict)
#build confusion matrix
RF_CM = pd.DataFrame(confusion_matrix(Y_test, RF_Predict))
RF_Accuracy, RF_FNR = validationMatrix(RF_CM)
print("Random Forest confusion matrix\n", RF_CM)
print("\nRandom Forest Accuracy",RF_Accuracy)
print("Random Forest False Negative rate",RF_FNR)

# Logistic Regression
logit_data = churn_data.copy()
#Create logistic data. Save target variable first
churn_data_logit = pd.DataFrame(logit_data['Churn'])
churn_data_logit = churn_data_logit.join(logit_data[numeric_cnames])
#Create dummies for categorical variables
cat_names = ["state", "international plan", "voice mail plan", "number customer service calls"]
for i in cat_names:
    temp = pd.get_dummies(logit_data[i], prefix = i)
    churn_data_logit = churn_data_logit.join(temp)
logit_train = churn_data_logit.iloc[:3333, :]
logit_test = churn_data_logit.iloc[3333:, :]
#Divide data into train and test
X_logit_train = logit_train.values[:, 0:14]
Y_logit_train = logit_train.values[:,14]
X_logit_test = logit_test.values[:, 0:14]
Y_logit_test = logit_test.values[:,14]
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression()
LR_model.fit(X_logit_train, Y_logit_train)
LR_predict = LR_model.predict(X_logit_test)
accuracy_score(Y_logit_test, LR_predict)
#build confusion matrix
LR_CM = pd.DataFrame(confusion_matrix(Y_logit_test, LR_predict))
LR_Accuracy, LR_FNR = validationMatrix(LR_CM)
print("Logistic Regression confusion matrix\n", LR_CM)
print("\nLogistic Regression Accuracy",LR_Accuracy)
print("Logistic Regression False Negative rate",LR_FNR)

# KNN implementation
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors = 7).fit(X_train, Y_train)
#predict test cases
KNN_Predict = KNN_model.predict(X_test)
accuracy_score(Y_test, KNN_Predict)
#build confusion matrix
KNN_CM = pd.DataFrame(confusion_matrix(Y_test, KNN_Predict))
KNN_Accuracy, KNN_FNR = validationMatrix(KNN_CM)
print("KNN confusion matrix\n", KNN_CM)
print("\nKNN Accuracy",KNN_Accuracy)
print("KNN False Negative rate",KNN_FNR)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
#Naive Bayes implementation
NB_model = GaussianNB().fit(X_train, Y_train)
#predict test cases
NB_Predict = NB_model.predict(X_test)
accuracy_score(Y_test, NB_Predict)
#Build confusion matrix
NB_CM = pd.DataFrame(confusion_matrix(Y_test, NB_Predict))
NB_Accuracy, NB_FNR = validationMatrix(NB_CM)
print("Naïve Bayes confusion matrix\n", NB_CM)
print("\nNaïve Bayes Accuracy",NB_Accuracy)
print("Naïve Bayes False Negative rate",NB_FNR)
