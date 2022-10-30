#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction Linear Regression Model

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[10]:


house_price_dataset = sklearn.datasets.load_boston()


# In[12]:


print(house_price_dataset)


# In[13]:


house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)
#numpy into pandas dataframe


# In[15]:


house_price_dataframe ['Price'] = house_price_dataset.target
#adding the price column (target column) to the data frame


# In[16]:


house_price_dataframe.head()


# In[ ]:


# Attribute infromation
# CRIM     per capita crime rate by town
# ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS    proportion of non-retail business acres per town
# CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX      nitric oxides concentration (parts per 10 million)
# RM       average number of rooms per dwelling
# AGE      proportion of owner-occupied units built prior to 1940
# DIS      weighted distances to five Boston employment centres
# RAD      index of accessibility to radial highways
# TAX      full-value property-tax rate per $10,000
# PTRATIO  pupil-teacher ratio by town
# B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
# LSTAT    % lower status of the population
# MEDV     Median value of owner-occupied homes in $100


# In[19]:


house_price_dataframe.isnull().sum() #Checking missing values


# In[39]:


#Separating the target column andother columns


# In[21]:


X = house_price_dataframe.drop(['Price'],axis = 1)
Y = house_price_dataframe['Price']


# In[22]:


print(X)
print(Y)


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[25]:


#Loading the Model
Linear_model = LinearRegression()


# In[27]:


#Tarining the model
Linear_model.fit(X_train,Y_train)


# In[29]:


#Accuracy on prediction on training data
training_data_prediction = Linear_model.predict(X_train)


# In[30]:


#R square error
score_1 = metrics.r2_score(Y_train,training_data_prediction)


# In[31]:


print('R square error :',score_1)


# R-Squared (R² or the coefficient of determination) is a statistical measure in a regression model that determines the proportion of variance in the dependent variable that can be explained by the independent variable. In other words, r-squared shows how well the data fit the regression model (the goodness of fit)

# In[38]:


plt.scatter(Y_train,training_data_prediction)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted prices')
plt.title('Actual prices vs Predicted prices')
plt.show()


# In[ ]:




