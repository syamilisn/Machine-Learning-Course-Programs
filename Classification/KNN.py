from sklearn.datasets import load_iris

dataset = load_iris()
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts= train_test_split(X,y,train_size=0.25,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors=5, metric= 'minkowski',p=2)
clf = KNeighborsClassifier()
clf.fit(xtr, ytr)

print(" Accuracy=",clf.score(xts, yts))

print("Predicted Data")
print(clf.predict(xts))

prediction=clf.predict(xts)

print("Test data :")
print(yts)

#6 To identify the miss classification
diff=prediction-yts
print("Result is ")
print(diff)
print('Total no of samples misclassied =', sum(abs(diff)))