import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('Datasets\Salary_Data.csv')
#print(dataset)
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2, random_state=1)
#print(xtrain)
#print(xtest)
#print(ytrain)
#print(ytest)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)