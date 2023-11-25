import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv("/content/diabetes.csv")
df
x = df.iloc[:,0:8].values
x
y = df.iloc[:,-1].values
y
from sklearn .model_selection import train_test_split
test_x
test_y
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
clf.fit(x, y)
clf.predict([[6,148,72,35,0,33.6,0.627,50]])
//to find the accuracy of the model
MNIST_df = pd.read_csv("/content/digit_svm.csv")
MNIST_df
t = MNIST_df.iloc[:,1:].values
t
x = np.nan_to_num(t)
x
y = MNIST_df.iloc[:,0].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2, random_state = 4, stratify=y)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = SVC (C= 1 , kernel='linear')
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
