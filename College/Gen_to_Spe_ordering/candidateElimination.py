import numpy as np
import pandas as pd

dataset = pd.read_csv('B:\AI ML\AI-ML-Lab-Programs\Pg3\Training_examples.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

def learn(X, y):
    n = len(X[0])
    print("The no. of attbs:",n)
    print("Initialization of specific_h and general_h:")
    s_ht = X[0].copy()
    print("Initial Specific h/t:\n", s_ht)
    g_ht = [["?" for i in range(n)] for i in range(n)]
    print("Initial Generic h/t:")
    print(g_ht)

    for i,h in enumerate(X):        #enumerate() method adds counter to an iterable and returns it.
        if y[i] == 'Yes':           #generalization
            for ab in range(n):
                if h[ab] != s_ht[ab] :
                    s_ht[ab] = '?'
                    g_ht[ab][ab] = '?'

        if y[i] == 'No':            #specialization
            for ab in range(n):
                if h[ab] != s_ht[ab] :                   
                    g_ht[ab][ab] = s_ht[ab]
                else:
                    g_ht[ab][ab] = '?'
        print(" Steps of Candidate Elimination Algorithm",i+1)
        print("Specific h/t:\n",s_ht)
        print("General h/t:\n",g_ht)
    
    unused_ht = ['?', '?', '?', '?', '?', '?']      # find indices where we have empty rows, meaning those that are unchanged
    while g_ht.count(unused_ht):
        g_ht.remove(unused_ht)
    return s_ht,g_ht

s_final,g_final=learn(X,y)
print("Final Specific_h:\n", s_final)
print("Final General_h:\n", g_final)