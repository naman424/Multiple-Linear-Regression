#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
train=pd.read_csv('Train.csv')
X_test=pd.read_csv('Test.csv')
X_train=train.iloc[:,:-1].values
y_train=train.iloc[:,-1].values

#multiple regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#predicting values of the test set
y_pred=regressor.predict(X_test)

#using backward elimination to optimise the model 
import statsmodels.regression.linear_model as sm
X_train=np.append(arr=np.ones((1600,1)),values=X_train,axis=1)
X_opt=X_train[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y_train,exog=X_opt).fit()
regressor_OLS.summary()   #since the p-value for all the variables is zero hence all 
#the variables are equally import and we cannot remove any more variables as it will 
#lead to a less reliable model
