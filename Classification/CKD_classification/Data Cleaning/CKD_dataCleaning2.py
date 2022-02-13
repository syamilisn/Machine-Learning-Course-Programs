from matplotlib.cbook import index_of
import numpy as np
import pandas as pd
from write_csv import wc
dataset = pd.read_csv('Datasets\CKD_smol.csv')
names=['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane','class']
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
data = dataset.iloc[:,:].values
row = len(dataset)
col = len(X[0])
print('no. rows',row)
print('no. cols',col)

#replace ?
for i in range(row):
    for j in range(col):
        if X[i][j] == '?':
            X[i][j] = np.nan
print(X[0])

#Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,:])
data[:,:-1] = imputer.transform(X[:,:])
print("New record: ",data[0])
print("Old record: ",X[0])
filename = 'CKD_cleaned2'
wc.write_csv(data,names,filename)