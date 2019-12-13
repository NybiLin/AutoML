#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
csv = "VEND.csv"
csv_test = "VENDTEST.csv"
Y = "SalePrice"
class Automl:
    def __init__(self, csv, target, reg_or_class):
        self.df = pd.read_csv(csv)
        self.shape = self.df.shape
        self.target = target
        self.type = reg_or_class
        self.score = self.predict()
    
    def clean(self):
        self.df = self.df.dropna(thresh= 1*len(self.df), axis=1) #Drop na selon le threshold4
        #sépare cat et num
        self.categorical = self.df.select_dtypes(exclude=['number'])
        self.numerical = self.df.select_dtypes(include=['number'])
        list_num = list(self.numerical.columns)
        #normalise les donnes
        self.categorical_d = pd.get_dummies(self.categorical, drop_first=True)

        scaler = StandardScaler()
        scaler.fit(self.numerical)
        self.numerical = scaler.transform(self.numerical)
        self.numerical = pd.DataFrame(self.numerical,  columns= list_num)
        #concat le DF num et le DF cat apres standardisation et normalisation
        self.df_clean = pd.concat([self.categorical_d, self.numerical], axis=1)
        #Drop le colonne ID si elle existe

        self.X = self.df_clean.drop(['Id'], axis=1).values
        self.y = self.df_clean[self.target].values
        #Drop le y du df principal
        if self.type !="linear":
            self.y[self.y>0] = 1
            self.y[self.y<=0] = 0
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=0.2, random_state=69)
    
    def lin_reg(self):
        self.reg = LinearRegression().fit(self.X_train, self.y_train) 
        self.y_pred = self.reg.predict(self.X_test)
        return self.reg.score(self.X_test,self.y_test)
    
    def log_reg(self):
        self.reg = LogisticRegression(solver='lbfgs',max_iter=1000).fit(self.X_train, self.y_train)
        self.yreg_pred = self.reg.predict(self.X_test)
        return self.reg.score(self.X_test,self.y_test)
    
    def predict(self):
        self.clean()
        if self.type == 'linear':
            score = self.lin_reg()
        elif self.type == 'logistic':
            score = self.log_reg()
        return "predicted score : ", score
    
    def cross_val(self):
        self.clean()
        if self.type == 'linear':
            score = self.lin_reg()
        elif self.type == 'logistic':
            score = self.log_reg()
        regressor = DecisionTreeRegressor(random_state=0)
        val = cross_val_score(self.reg, self.X_test, self.y_test, cv=120)
        val = np.mean(val)
        return "Cross validation score : ", val
    

