# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets
2. Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters
3. Train your model -Fit model to training data -Calculate mean salary value for each subset
4. Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance
5. Tune hyperparameters -Experiment with different hyperparameters to improve performance
6. Deploy your model Use model to make predictions on new data in real-world application.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Vanisha Ramesh
RegisterNumber:  212222040174

import pandas as pd
df=pd.read_csv('/content/Salary.csv')

df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()

x=df[["Position","Level"]]
y=df["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
*/
```

## Output:
1.data.head()

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119104009/0d168976-fcc6-4195-8ac3-ce5ac2aab830)

2.data.info()

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119104009/3bd8abf4-8568-4a82-966d-923852fac25a)

3.data.isnull().sum()

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119104009/84394276-95e6-43b5-8283-94b8c2f3c9c2)

4.data.head() for position:

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119104009/7625904c-44e8-427b-ac73-e53567ea5ea7)

5. MSE value:

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119104009/7cb439b6-5d83-4b54-b3ce-897142f237bd)


6.R2 value:

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119104009/1f7fc08f-12f7-4d7a-b060-e374a99b02c9)

7.Prediction Value:

![image](https://github.com/Vanisha0609/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119104009/ae519a28-03a4-4f7c-b97d-c99a5fc49c75)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
