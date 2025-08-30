## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/
```
```   
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv("K:\downloads\student_scores.csv")
print(dataset.head())
print(dataset.tail())

x = dataset.iloc[:, :-1].values
print(x)
y = dataset.iloc[:, 1].values
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train, y_train, color='yellow')
plt.plot(x_train, reg.predict(x_train), color='red')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test, y_test, color='black')
plt.plot(x_train, reg.predict(x_train), color='darkgreen')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_absolute_error(y_test, y_pred)
print('Mean Square Error = ', mse)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error = ', mae)
rmse = np.sqrt(mse)
print("Root Mean Square Error = ", rmse)

```


## Output:

<img width="1073" height="813" alt="Screenshot 2025-08-30 143404" src="https://github.com/user-attachments/assets/ceb83aed-10b0-4e4e-8472-f23c5e184e7f" />

<img width="704" height="526" alt="Screenshot 2025-08-30 143423" src="https://github.com/user-attachments/assets/8b731fff-b4a6-4b1a-966e-95a99d25dbf5" />

<img width="704" height="511" alt="Screenshot 2025-08-30 143416" src="https://github.com/user-attachments/assets/b131c729-cef8-4afc-af89-31ed04f28c57" />

<img width="436" height="78" alt="Screenshot 2025-08-30 143427" src="https://github.com/user-attachments/assets/b27c840c-7cfb-4de2-a4ec-07cd816a0de4" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
