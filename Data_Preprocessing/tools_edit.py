import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Datasets\Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(X)
print(y)

#Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3 ])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)

#encoding categorical data (1. Independent variable 2. Dependent variable)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough') #0 here is the indeX of the col to be transformed
X = ct.fit_transform(X)
print(X)

#2 categories: 1/0
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y = le.fit_transform(y)
print(y)

#Splitting into training and test set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=1)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain[:,3:] = sc.fit_transform(xtrain[:,3:])
xtest[:,3:] = sc.transform(xtest[:,3:])
print(xtrain)
print(xtest)