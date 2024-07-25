#datapreprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('50_Startups.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#Encoding the categorical data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-1])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

#splitting data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
#print(X_train)
#print(y_train)

#training the data

from sklearn.linear_model import LinearRegression 
regressor= LinearRegression()
regressor.fit(X_train,y_train)

#predicting

y_pred=regressor.predict(X_test)
print(y_pred.reshape(len(y_pred),1))
np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_pred),1)),1))

#predicting for specific values 
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))
#getting the equation
print(regressor.coef_)
print(regressor.intercept_)