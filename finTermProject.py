#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:16:04 2018
CS 634: Data Mining
Final Term Project : Logistic Regression and Random Forest Comparison
@author: LWC
"""

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.feature_selection
from sklearn.feature_selection import chi2
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.metrics import classification_report

##############################################################################

filename = 'Adult_DataSet_Complete.csv'

column_names = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-Num',
			  'Marital-Status', 'Occupation', 'Relationship', 'Race', 
                'Sex', 'Capital-Gain', 'Capital-Loss', 'Hours-Per-Week', 
                'Native-Country', 'Income']
				
data = pd.read_csv(filename,
                   sep=',',
                   skiprows=[0],
                   header=None,
                   names=column_names,
                   engine='python')
			 	   

#print(data['Income'].value_counts())
data['Income'].replace([' <=50K.', ' >50K.'], [' <=50K', ' >50K'], inplace=True)
#print(data['Income'].value_counts())


############################## Observe the Data ##############################
#data.shape
#data.head(10)
#print(data[['Income', 'Age', 'Fnlwgt', 'Education-Num',
#            'Capital-Gain', 'Capital-Loss',
#            'Hours-Per-Week']].groupby('Income').std())
#print('\n')
#print(data[['Income', 'Age', 'Fnlwgt', 'Education-Num',
#            'Capital-Gain', 'Capital-Loss',
#            'Hours-Per-Week']].groupby('Income').mean())
#print('\n')
#print(data[['Income', 'Age', 'Fnlwgt', 'Education-Num',
#            'Capital-Gain', 'Capital-Loss',
#            'Hours-Per-Week']].groupby('Income').median())
#print('\n')
#print(data[['Income', 'Age', 'Fnlwgt', 'Education-Num',
#            'Capital-Gain', 'Capital-Loss',
#            'Hours-Per-Week']].groupby('Income').min(skipna=True))
#print('\n')
#print(data[['Income', 'Age', 'Fnlwgt', 'Education-Num',
#            'Capital-Gain', 'Capital-Loss',
#            'Hours-Per-Week']].groupby('Income').max(skipna=True))

##############################################################################

X = data.drop('Income', 1)
Y = [0 if x == ' <=50K' or x == ' <=50K.' else 1 for x in data['Income']]

#### Continuous Values #### Get Mean, Median, Standard Deviation, Min, Max
#### HISTOGRAM ####
def get_stats(x):
    StanDev = x.std(skipna=True)
    Median = x.median(skipna=True)
    Mean = x.mean(skipna=True)
    Min = x.min(skipna=True)
    Max = x.max(skipna=True)
    return [StanDev, Median, Mean, Min, Max]

##print(X['Age'].value_counts().sort_values(ascending=False).head())
#Age_Stats = get_stats(X['Age'])
#print("Stats - Feature: Age")
#print("Standard Deviation: {} \nMedian: {} \nMean: {} \nMin: {} \nMax: {}".format(
#        Age_Stats[0], Age_Stats[1], Age_Stats[2],
#        Age_Stats[3], Age_Stats[4]))
#
##print(X['Education-Num'].value_counts().sort_values(ascending=False).head())
#Ed_Num_Stats = get_stats(X['Education-Num'])
#print("Stats - Feature: Education-Num")
#print("Standard Deviation: {0} \nMedian: {1} \nMean: {2} \nMin: {3} \nMax: {4}".format(
#      Ed_Num_Stats[0], Ed_Num_Stats[1], Ed_Num_Stats[2], 
#      Ed_Num_Stats[3], Ed_Num_Stats[4]))  
#
##print(X['Hours-Per-Week'].value_counts().head())
#HPW_Stats = get_stats(X['Hours-Per-Week'])
#print("Stats - Feature: Hours-Per-Week")
#print("Standard Deviation: {0} \nMedian: {1} \nMean: {2} \nMin: {3} \nMax: {4}".format(
#      HPW_Stats[0], HPW_Stats[1], HPW_Stats[2], 
#      HPW_Stats[3], HPW_Stats[4]))
#
##print(X['Fnlwgt'].value_counts().head())
#FinalWeight_Stats = get_stats(X['Fnlwgt'])
#print("Stats - Feature: Fnlwgt")
#print("Standard Deviation: {0} \nMedian: {1} \nMean: {2} \nMin: {3} \nMax: {4}".format(
#      FinalWeight_Stats[0], FinalWeight_Stats[1], FinalWeight_Stats[2], 
#      FinalWeight_Stats[3], FinalWeight_Stats[4]))
#
##print(X['Capital-Gain'].value_counts().head())
#CapGain_Stats = get_stats(X['Capital-Gain'])
#print("Stats - Feature: Capital-Gain")
#print("Standard Deviation: {0} \nMedian: {1} \nMean: {2} \nMin: {3} \nMax: {4}".format(
#      CapGain_Stats[0], CapGain_Stats[1], CapGain_Stats[2], 
#      CapGain_Stats[3], CapGain_Stats[4]))
#
##print(X['Capital-Loss'].value_counts().head())
#CapLoss_Stats = get_stats(X['Capital-Loss'])
#print("Stats - Feature: Capital-Loss")
#print("Standard Deviation: {0} \nMedian: {1} \nMean: {2} \nMin: {3} \nMax: {4}".format(
#      CapLoss_Stats[0], CapLoss_Stats[1], CapLoss_Stats[2], 
#      CapLoss_Stats[3], CapLoss_Stats[4]))

	
#### Discrete Values ####
#### Bar ####
#print(X['Workclass'].value_counts() / len(X.index))
#print(X['Education'].value_counts() / len(X.index))
#print(X['Marital-Status'].value_counts() / len(X.index))	
#print(X['Occupation'].value_counts() / len(X.index))
#print(X['Relationship'].value_counts() / len(X.index))
#print(X['Race'].value_counts() / len(X.index))
#print(X['Sex'].value_counts() / len(X.index))
#print(X['Native-Country'].value_counts() / len(X.index))	

##############################################################################
####     G R A P H S    
##############################################################################

def plot_histogram(x):
    plt.hist(x, color='blue', range=(x.min(), x.max()), bins=15, edgecolor='black', alpha=0.5)	
    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
 
#plot_histogram(X['Hours-Per-Week'])
#plot_histogram(X['Age'])
#plot_histogram(X['Education-Num'])
#plot_histogram(X['Fnlwgt'])
#plot_histogram(X['Capital-Gain'])
#plot_histogram(X['Capital-Loss'])


def plot_categorical(x, kin):
    # kinds [line, bar, barh, hist, box, kde, density, area, pie, scatter, hexbin]
    data = (x.value_counts() / len(x.index))
    data.plot(kind=kin, edgecolor='black', alpha=0.5)
    plt.title("Percentage Frequency of '{var}'".format(var=x.name))
    plt.show()
    
#plot_categorical(X['Occupation'], 'barh')
#plot_categorical(X['Workclass'], 'barh')
#plot_categorical(X['Education'], 'barh')
#plot_categorical(X['Marital-Status'], 'barh')
#plot_categorical(X['Relationship'], 'barh')
#plot_categorical(X['Race'], 'barh')
#plot_categorical(X['Sex'], 'barh')
#plot_categorical(X['Native-Country'], 'bar') 
    
 
def plot_comparison(x):
    pd.crosstab(x, Y).plot(kind='barh')
    plt.title("Income Frequency for '{var_name}'".format(var_name=x.name))
    plt.xlabel("Frequency")
    plt.ylabel("Frequency")
    plt.show()
    
#plot_comparison(data['Occupation'])
#plot_comparison(data['Workclass'])
#plot_comparison(data['Education'])
#plot_comparison(data['Marital-Status'])
#plot_comparison(data['Relationship'])
#plot_comparison(data['Race'])
#plot_comparison(data['Sex'])
#plot_comparison(data['Native-Country'])    


##########################################################################
####        P R E - P R O C E S S I N G    
##########################################################################  


#### To group low frequency categories in Native-Country Variable
X['Native-Country'] = ['United-States' 
                        if x == ' United-States' else 'Other' 
                        for x in X['Native-Country']]

X['Native-Country'].unique()
X['Native-Country'].value_counts()

#### To group low frequency categories in Race Variable
X['Race'] = ['White' 
             if x == ' White' else 'Non-White' 
             for x in X['Race']]

X['Race'].unique()
X['Race'].value_counts()

#### Function to group categories in Workclass Variable
def workclass_groups(x):
    if x in [' Private', ' Self-emp-not-inc', ' Self-emp-inc']:
        return 'Private'
    elif x in [' Local-gov', ' State-gov', ' Federal-gov']:
        return 'Public'
    else:
        return 'Other' 

X['Workclass'] = X['Workclass'].apply(workclass_groups)
X['Workclass'].value_counts()
X['Workclass'].unique()

#### Function to group categories in Education Variable
def education_groups(x):
    if x in [' HS-grad', ' 9th', ' 10th', ' 11th', ' 12th']:
        return 'Highschool'
    elif x in [' Some-college', ' Bachelors', ' Assoc-voc', ' Assoc-acdm']:
        return 'College'
    elif x in [' Masters', ' Prof-school', ' Doctorate']:
        return 'Grad-School'
    else:
        return 'Other' 

X['Education'] = X['Education'].apply(education_groups)
X['Education'].value_counts()
X['Education'].unique()

#### Function to group categories in Marital-Status Variable
def marriage_groups(x):
    if x in [' Married-civ-spouse', ' Married-spouse-absent', 
             ' Separated', ' Married-AF-spouse']:
        return 'Married'
    else:
        return 'Unmarried' 

X['Marital-Status'] = X['Marital-Status'].apply(marriage_groups)
X['Marital-Status'].value_counts()
X['Marital-Status'].unique()


#### DROP Education Number ####
X = X.drop('Education-Num', axis=1)

#### List of features to dummy - convert to 1's and 0's
feature_list = ['Workclass', 'Education', 'Marital-Status', 'Occupation',
   				'Relationship', 'Race', 'Sex', 'Native-Country']

#### Function to dummy categorical variables 
def dummy(dataframe, features):
    for feature in features:
        dummies = pd.get_dummies(dataframe[feature], prefix=feature)
        dataframe = dataframe.drop(feature, axis=1)
        dataframe = pd.concat([dataframe, dummies], axis=1)
    return dataframe
 	
X_dum = dummy(X, feature_list)
#X_dum.dtypes.value_counts()


##########################################################################
####        F E A T U R E   A D D I N G
##########################################################################

def add_interactions(data):
	combos = list(combinations(list(data.columns), 2))
	var_names = list(data.columns) + ['_'.join(x) for x in combos]
	poly = PolynomialFeatures(interaction_only=True, include_bias=False)
	data = poly.fit_transform(data)
	data = pd.DataFrame(data)
	data.columns = var_names
	
	return data
	
X_add = add_interactions(X_dum)
#X_add.dtypes

##########################################################################
####        F E A T U R E   S E L E C T I O N
##########################################################################

####    No added features
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(
        X_dum, Y, test_size=0.3, train_size=0.70, random_state=1)

####    With added features
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
        X_add, Y, test_size=0.3, train_size=0.70, random_state=1)

####    Reduce added features by selecting the top K
select = sklearn.feature_selection.SelectKBest(chi2, k=40)
selected_features = select.fit(X_train2, Y_train2)
selected_indexes = selected_features.get_support(indices=True)
selected_cols = [X_add.columns[i] for i in selected_indexes]

X_train_selected = X_train2[selected_cols]  
X_test_selected = X_test2[selected_cols]


##########################################################################
####        M O D E L   B U I L D I N G
##########################################################################

###############  LOGISTIC REGRESSION  ##################

#### Labels ####
def performance1(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    score1 = accuracy_score(Y_test, y_pred, normalize=True)
    return score1

#### Probabilities ####
def performance2(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    y_hat = model.predict_proba(X_test)[:,1]
    score = roc_auc_score(Y_test, y_hat)
    return score

model_LR = performance1(X_train1, Y_train1, X_test1, Y_test1)
print("LOGISTIC REGRESSION - Accuracy Score of feature category grouping"
      "\n\twithout interactions and feature selection (labelling): {:6.4f}" 
      .format(model_LR))

model_LR_AUC = performance2(X_train1, Y_train1, X_test1, Y_test1)
print("LOGISTIC REGRESSION - ROC_AUC Score of feature category grouping"
      "\n\twithout interactions and feature selection (Continuous Values): " 
      "{:6.4f}".format(model_LR_AUC))

model_LR2 = performance1(X_train_selected, Y_train2, X_test_selected, Y_test2)
print("LOGISTIC REGRESSION - Accuracy Score of feature category grouping"
      "\n\twith interactions and feature selection (labelling): {:6.4f}" 
      .format(model_LR2))

model_LR_AUC2 = performance2(X_train_selected, Y_train2,
                             X_test_selected, Y_test2)
print("LOGISTIC REGRESSION - ROC_AUC Score of feature category grouping"
      "\n\twith interactions and feature selection (Continuous Values): "
      "{:8.6f}".format(model_LR_AUC2))

model0 = LogisticRegression()
model0.fit(X_train1, Y_train1)
y0_pred = model0.predict(X_test1)
y_hat0 = model0.predict_proba(X_test1)[:,1]
score0 = roc_auc_score(Y_test1, y_hat0)

model1 = LogisticRegression()
model1.fit(X_train_selected, Y_train1)
y1_pred = model1.predict(X_test_selected)
y_hat = model1.predict_proba(X_test_selected)[:,1]
score1 = roc_auc_score(Y_test2, y_hat)

###############  RANDOM FOREST  ##################

model2 = RandomForestClassifier()
model2.fit(X_train_selected, Y_train2)
y2_pred = model2.predict(X_test_selected)
y_hat2 = model2.predict_proba(X_test_selected)[:,1]
score2 = roc_auc_score(Y_test2, y_hat2)

model3 = RandomForestClassifier()
model3.fit(X_train1, Y_train1)
y3_pred = model3.predict(X_test1)
y_hat3 = model3.predict_proba(X_test1)[:,1]
score3 = roc_auc_score(Y_test1, y_hat3)

print("RANDOM FOREST - Accuracy Score of feature category grouping with"
      "\n\tinteractions and feature selection (Continuous Values): {:8.6f}" 
      .format(accuracy_score(Y_test2, y2_pred)))

print("RANDOM FOREST - ROC_AUC Score of feature category grouping with"
      "\n\tinteractions and feature selection (Continuous Values): {:8.6f}" 
      .format(score2))

print("RANDOM FOREST - Accuracy Score of feature category grouping without"
      "\n\tinteractions and feature selection (Continuous Values): {:8.6f}" 
      .format(accuracy_score(Y_test1, y3_pred)))

print("RANDOM FOREST - ROC_AUC Score of feature category grouping without"
      "\n\tinteractions and feature selection (Continuous Values): {:8.6f}" 
      .format(score3))

#### Comparing Models ####
print('Logistic Regression Model W/O interactions & selection: \n'
      + classification_report(Y_test1, y0_pred))
print('Logistic Regression Model WITH interactions & selection: \n'
      + classification_report(Y_test2, y1_pred))
print('Random Forest Model WITH interactions & selection: \n'
      + classification_report(Y_test2, y2_pred))
print('Random Forest Model W/O interactions & selection: \n'
      + classification_report(Y_test1, y3_pred))

##########################################################################
####       R O C   C U R V E
##########################################################################


#### ALL GRAPHS ####    
fpr0, tpr0, thresh0 = roc_curve(Y_test1, y_hat0)
s0 = roc_auc_score(Y_test1, y_hat0) 
fpr1, tpr1, thresh1 = roc_curve(Y_test2, y_hat)
s1 = roc_auc_score(Y_test2, y_hat)   
fpr2, tpr2, thresh2 = roc_curve(Y_test2, y_hat2)
s2 = roc_auc_score(Y_test2, y_hat2) 
fpr3, tpr3, thresh3 = roc_curve(Y_test1, y_hat3)
s3 = roc_auc_score(Y_test1, y_hat3) 

plt.figure()
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='By Chance')
plt.plot(fpr0, tpr0, label='Logistic Regression model 1 (area = %0.2f)' % (s0))
plt.plot(fpr1, tpr1, label='logistic Regression modle 2 (area = %0.2f)' % (s1))
plt.plot(fpr2, tpr2, label='Random Forest model 3 (area = %0.2f)' % (s2))
plt.plot(fpr3, tpr3, label='Random Forest model 4 (area = %0.2f)' % (s3))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_Curve_ALL_MODELS')
plt.show()

##########################################################################
####       C R O S S   V A L I D A T I O N
##########################################################################


kfold = model_selection.KFold(n_splits=10, shuffle=False, random_state=None)

#### ROC AUC ####
results1 = model_selection.cross_val_score(model0,
                                          X_dum,
                                          Y,
                                          cv=kfold,
                                          scoring='roc_auc')

results2 = model_selection.cross_val_score(model1,
                                          X_add[selected_cols],
                                          Y,
                                          cv=kfold,
                                          scoring='roc_auc')

results3 = model_selection.cross_val_score(model2,
                                          X_add[selected_cols],
                                          Y,
                                          cv=kfold,
                                          scoring='roc_auc')

results4 = model_selection.cross_val_score(model3,
                                          X_dum,
                                          Y,
                                          cv=kfold,
                                          scoring='roc_auc')

print(results1.mean())
print(results2.mean())
print(results3.mean())
print(results4.mean())

#### Accuracy ####
results5 = model_selection.cross_val_score(model0,
                                          X_dum,
                                          Y,
                                          cv=kfold,
                                          scoring='accuracy')

results6 = model_selection.cross_val_score(model1,
                                          X_add[selected_cols],
                                          Y,
                                          cv=kfold,
                                          scoring='accuracy')

results7 = model_selection.cross_val_score(model2,
                                          X_add[selected_cols],
                                          Y,
                                          cv=kfold,
                                          scoring='accuracy')

results8 = model_selection.cross_val_score(model3,
                                          X_dum,
                                          Y,
                                          cv=kfold,
                                          scoring='accuracy')

print(results5.mean())
print(results6.mean())
print(results7.mean())
print(results8.mean())

