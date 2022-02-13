import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Datasets\CKD_smol.csv')
names=['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane','class']
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print("X[0]",X[0])
print("y[0]",y[0])
row = len(dataset)
col = len(X[0])
print('no. rows',row)
print('no. cols',col)
#Converting ? to NaN
print("Each record:")

for i in range(5):
    print(f"Row {i+1}:")
    for j in range(10):
        print(end=' ')
        if i<=10:
            print(f"{names[j]}:",X[i][j],end='')
    print()

for i in range(row):
    for j in range(col):
        if X[i][j] == '?':
            X[i][j] = np.nan
print(X[0])

#Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,:])
X[:,:] = imputer.transform(X[:,:])
print(X[0])

#display row and col no.
for row, col in enumerate(X):
    print(row,col)
