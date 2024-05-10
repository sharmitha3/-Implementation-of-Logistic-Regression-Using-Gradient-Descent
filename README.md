# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
~~~
1. Import the necessary libraries: pandas, numpy, and matplotlib.pyplot.
2. Read the dataset using pandas read_csv function.
3. Preprocess the data: drop unnecessary columns, convert categorical variables to numerical using label encoding, and separate features (X) and target (Y) variables.
4. Define the logistic regression model functions: sigmoid function, loss function, gradient descent function, and prediction function.
5. Generate random data for training
6. Train the logistic regression model using gradient descent with the generated or provided data.
7.Evaluate the trained model: calculate the accuracy on the training data.
~~~
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHARMITHA V
RegisterNumber: 212223110048
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Placement_Data (1).csv')
data
data=data.drop("sl_no",axis=1)
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data.dtypes
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_S"]=data["hsc_s"].cat.codes
data
X=data.iloc[:, :-1].values
Y=data.iloc[:, -1].values
Y
import numpy as np
X = np.random.randn(100, 5)
Y = np.random.randint(0, 2, size=(100,))  
X = np.array(X)
y = np.array(Y)

theta = np.random.randn(X.shape[1])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0]]) 

y_prednew = predict(theta, xnew)
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0]])
y_prednew = predict(theta, xnew)
print(y_prednew)
*/
```

## Output:
![image](https://github.com/sharmitha3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145974496/14b0f61d-3e66-4eca-85fe-90dd93d7d1df)
![image](https://github.com/sharmitha3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145974496/6a5098c6-43c4-4d86-9535-d54930456dac)
![image](https://github.com/sharmitha3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145974496/8555e21b-db39-41ba-8f5e-f04f352ef51e)
![image](https://github.com/sharmitha3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145974496/30abaa6b-e491-4fbb-8613-31f33b60d35c)
![image](https://github.com/sharmitha3/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145974496/a5ad0e85-9706-4e64-914d-f0091c2a0afd)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

