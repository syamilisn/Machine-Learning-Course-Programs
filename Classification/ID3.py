import pandas as pd
import numpy as np

dataset = pd.read_csv('B:\AI ML\AI-ML-Lab-Programs\pg4\playtennis1.csv',names=['outlook','temperature','humidity','wind','class'])

def entropy(y):         #y = target column
    vals,counts = np.unique(y,return_counts = True)
    print("the len is ",counts)
    total_count = np.sum(counts)
    entropy = np.sum([ (-counts[i]/total_count) * np.log2(counts[i]/total_count) for i in range(len(vals)) ])
    return entropy

def InfoGain(data,x,yname):
    total_entropy = entropy(data[yname])
    vals,counts= np.unique(data[x],return_counts=True)
    Weighted_Entropy = np.sum([ (counts[i]/np.sum(counts)) * entropy(data.where(data[x]==vals[i]).dropna()[yname]) for i in range(len(vals)) ])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain
    
def ID3(data,OGdata,X,yname="class",Pnode = None):      #X:features x: single feature Pnode: parent node class
    a = np.unique(data[yname])
    aT = np.unique(data[yname], return_counts=True)
    b = np.unique(OGdata[yname])
    bT = np.unique(OGdata[yname], return_counts=True)
    if len(a) <= 1:
        return a[0]
    elif len(data) == 0:
        return b[np.argmax(bT[1])]
    elif len(X) == 0:
        return Pnode
    else:
        Pnode = a[np.argmax(aT[1])]
        item_values = [InfoGain(data,x,yname) for x in X] #Return the information gain values for the features in the dataset
        best_x_index = np.argmax(item_values)
        best_x = X[best_x_index]
        tree = {best_x:{}}
        del(best_x_index) #deletes the feature stored in the particular index
        for value in np.unique(data[best_x]):
            sub_data = data.where(data[best_x] == value).dropna() #drops null values
            subtree = ID3(sub_data,dataset,X,yname,Pnode)
            tree[best_x][value] = subtree
        return(tree) 

training_data= dataset.iloc[:14].reset_index(drop=True)
print("Training data:\n",training_data)
tree = ID3(training_data,dataset,training_data.columns[:-1])
print('Display Tree',tree)
print('Length of training data=',len(training_data))