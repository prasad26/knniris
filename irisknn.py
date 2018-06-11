import numpy as np
import pandas as pd
from scipy import *
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#load the iris dataset as iris_dataset
iris_dataset = load.iris()
#Printing information about the loaded dataset
print("keys:\n{}",format(iris_dataset.keys()))
print(iris_dataset['target'])
print(iris_dataset['data'].shape)

#split the dataset into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(
	iris_dataset['data'], iris_dataset['target'], random_state = 0)

#save the dataset into pandas dataframe using pd.DataFrame
iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)


grid = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15),
                        marker='o', hist_kwds={'bins':20}, s=60, alpha=.8)

#Building prediction model for the dataset
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#test a sample input by injecting it manually
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)

#printing results of sample dataset
print("Prediction:\n{}".format(prediction))
print("Predicted target name:\n{}".format(iris_dataset['target_names'][prediction]))


#Evaluating the results using trained model
y_pred = knn.predict(X_test)

#printing the results of test dataset
print("Test set prediction:\n{}".format(y_pred))

#Testing the accuracy of model
print("Test set score:\n{}".format(knn.score(X_test, y_test)))









