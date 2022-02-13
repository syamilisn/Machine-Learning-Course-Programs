"""DATA PREPROCESSING TOOLS"""
#Import Libraries 
"""
numpy:arrays
matplotlib:graph
pandas:matrix and vectors
sklearn:machine learning models
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset using pandas fN
dataset = pd.read_csv('Datasets\Data.csv')
"""
dataset: variable containing matrix of csv file. 
var1: matrix of features vector
var2: dependent variable vector
iloc: locate indices (includes lower bound and excludes last col)
"""
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)
#Missing data
"""
(applicable to real numbers only)
1. Either delete the cells with the missing data or
2. Take average of the column values and replace in the missing data cell
"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3 ])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)

#encoding categorical data (1. Independent variable 2. Dependent variable)
"""
Where there are grouping, we want to encode them so that machine doesn't start interpreting pattern in them.
EG: country name
OneHotEncoder: In bg it creates a new column for each class(country) nad represents it as a binary vector.
France, germany, spain
001,010,100

These encoded col moves to 0th position
"""
#>2 categories
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough') #0 here is the index of the col to be transformed
x = ct.fit_transform(x)
print(x)

#2 categories: 1/0
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y = le.fit_transform(y)
print(y)

#Splitting into training and test set
"""
test_size: percentage of test set
random_state: selection of which tuples go into test set
"""
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=1)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

#Feature Scaling
"""
Reason: So that all features will be in same scale so that one feature won't dominate the other
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain[:,3:] = sc.fit_transform(xtrain[:,3:])
xtest[:,3:] = sc.transform(xtest[:,3:])
print(xtrain)
print(xtest)