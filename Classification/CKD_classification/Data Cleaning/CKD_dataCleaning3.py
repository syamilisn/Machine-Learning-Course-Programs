import numpy as np
import pandas as pd
from write_csv import wc
dataset = pd.read_csv('Datasets\CKD_Datasets\Chronic_Kidney_Disease.csv')
labels=['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane','class']
data = dataset.iloc[:,:].values
row = len(data)
col = len(data[0])

#replace ?
for i in range(row):
    for j in range(col):
        if data[i][j] == '?':
            data[i][j] = np.nan
print("Old record: ",data[0])

#Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
SimpleImputer()
imputer.fit(data[:,:-1])
data[:,:-1] = imputer.transform(data[:,:-1])
print("New record: ",data[0])

filename = 'CKD_cleaned_Final'
wc.write_csv(data,labels,filename)