import numpy as np
import pandas as pd
import tensorflow as tf
#Data preprocessing
dataset = pd.read_csv('Datasets/backProp.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Building ANN
def initialize_network():
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    return ann

def train_network():
    #compile
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    #train
    ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
    w = tf.get_variable("kernel")
    print(w.eval())

ann = initialize_network()
print(ann)