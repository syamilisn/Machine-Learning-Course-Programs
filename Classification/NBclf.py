import numpy as np
import pandas as pd

dataset = pd.read_csv('B:\AI ML\AI-ML-Lab-Programs\pg6\DBetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print('Test_data   Predicted_data')
prediction = [[ y_test[i],y_pred[i]] for i in range(len(y_pred))]
for i in prediction:
    print(i)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n",cm)
accuracy = 100 * accuracy_score(y_test, y_pred)
print("Accuracy is: \n%2.2f" % accuracy,"%")