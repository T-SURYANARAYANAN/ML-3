# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SURYANARAYANAN T
RegisterNumber:  212224040341
import pandas as pd
data=pd.read_csv(r"Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
## HEAD
![image](https://github.com/user-attachments/assets/1fa96c7d-9307-4559-86f4-197cd1d35aba)

## COPY
![image](https://github.com/user-attachments/assets/e36e33bc-08e8-461c-85d5-72cf63a4a269)

## FIT TRANSFORM
![image](https://github.com/user-attachments/assets/ac4e8694-cffd-4ea6-9614-cba2212e9b5d)

## LOGISTIC REGRESSION
![image](https://github.com/user-attachments/assets/9381e461-c29f-48fa-bce3-77c7ae7689a4)

## ACCURACY SCORE
![image](https://github.com/user-attachments/assets/df1a6dfd-3853-4d1f-b591-3effe574ee5d)

## CONFUSION MATRIX
![image](https://github.com/user-attachments/assets/e88f2c34-3ecb-42b4-bfde-3f5e3ca17860)

## CLASSIFICATION REPORT
![image](https://github.com/user-attachments/assets/4ee480e9-2adb-45ad-b7d4-ba78526a0b8b)

## PREDICTION
![image](https://github.com/user-attachments/assets/8d88101e-ffb7-47d4-9b68-622cb1a496f4)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
