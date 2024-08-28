# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data: Read the dataset using pd.read_csv() and display first and last few rows.
2.Prepare Data: Separate features (hours) and target variable (scores) for training and testing.
3.Split Data: Use train_test_split() to divide the dataset into training and testing sets.
4.Train Model: Fit a linear regression model using the training data.
5.Evaluate and Plot: Predict scores on the test set, and visualize results with scatter and line plots. 


## Program and Output:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:DHARANYA.N 
RegisterNumber:  212223230044
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
![image](https://github.com/user-attachments/assets/b347d2a5-5e3e-4446-8cbb-efb6d50c7a8f)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/4931df52-ecf6-4f12-8cea-e045a2df0ebd)
```
#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)
```
![image](https://github.com/user-attachments/assets/878f6c69-932e-474d-af91-cde5503ca377)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
```
```
X_train.shape,X_test.shape
```
![image](https://github.com/user-attachments/assets/130cf4c9-c46e-46ea-8220-07510b0733e8)
```
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
```
![image](https://github.com/user-attachments/assets/1c131973-e85b-46df-9337-3743637ff934)
```
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
![image](https://github.com/user-attachments/assets/89de8bd5-34b3-4044-99e1-a3f139133580)
```
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show
```
![image](https://github.com/user-attachments/assets/ba8acf3b-5682-4d42-ad6b-27b3047b03a2)
```
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show
```

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
