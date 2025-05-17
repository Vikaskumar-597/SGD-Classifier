# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Iris dataset and convert it into a Pandas DataFrame.
2.Split the DataFrame into features (X) and target (y).
3.Divide the data into training and testing sets.
4.Initialize an SGD classifier with specified parameters.
5.Train the classifier using the training data.
6.Predict the target values for the test data.
7.Calculate and display the accuracy and confusion matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by  : VIKASKUMAR M
RegisterNumber: 212224220122
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```
```
#Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()
```
```
X = df.drop('target', axis=1)
y = df['target']
#Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
```
```
#Creating an SGD classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
#Train the classifier on the training data
sgd_clf.fit(X_train, y_train)

```
```
#Make predictions on the testing data
y_pred = sgd_clf.predict(X_test)
#Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}%")
#Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
**Head Values-Dataframe**

![image](https://github.com/user-attachments/assets/21c46316-8c40-4f00-9481-f6b9c45ff77b)

**Fitting data on SGD Classifier**

![image](https://github.com/user-attachments/assets/198f6cac-3f31-4daf-981c-a4b94cd66574)

**Model Accuracy AND Confusion Matrix**

![image](https://github.com/user-attachments/assets/9fd3f174-63cb-434d-a403-2671c735289e)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
