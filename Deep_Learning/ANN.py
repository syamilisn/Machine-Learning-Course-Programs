import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

dataset = pd.read_csv('Datasets/Churn_Modelling.csv')
print(dataset)

x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values