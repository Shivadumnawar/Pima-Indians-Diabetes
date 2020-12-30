# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:29:12 2020

@author: shiva dumnawar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

df= pd.read_csv('diabetes.csv')

df.info()

df.describe()

df.isnull().sum()
# no null values

# data visualization

plt.figure(figsize=(8,6))
sns.countplot(x= 'Outcome', data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'Outcome', y='Pregnancies',  data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'Outcome', y='Glucose',  data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'Outcome', y='BloodPressure',  data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'Outcome', y='SkinThickness',  data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'Outcome', y='Insulin',  data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'Outcome', y='BMI',  data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'Outcome', y='DiabetesPedigreeFunction',  data= df)

plt.figure(figsize=(8,6))
sns.boxplot(x= 'Outcome', y='Age',  data= df)

plt.figure(figsize=(8,6))
plt.hist(x= df['Pregnancies'], bins= list(range(0,20)), edgecolor= 'black')

# check outliers
df.plot(kind='box', figsize= (14,8))
# there are outliers

df= df.clip(lower= df.quantile(0.1), upper= df.quantile(0.9), axis=1)

df.plot(kind='box', figsize= (14,8))
# no outliers

# correlation
plt.figure(figsize=(9,7))
c= df.corr()
sns.heatmap(c, cmap= 'coolwarm', annot=True, linewidth=0.25)
plt.tight_layout()

X= df.iloc[:, :-1]
y= df.iloc[:,-1]

# Stratified K Fold

from sklearn.model_selection import StratifiedKFold
skf= StratifiedKFold()

for train_index, test_index in skf.split(X,y):
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index],\
                                       y.iloc[train_index], y.iloc[test_index]
                                                                     
# scaling
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()

X_train= ss.fit_transform(X_train)
X_test= ss.fit_transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(X_train, y_train)

pred= knn.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_test, pred))

print(accuracy_score(y_test, pred))

# select the best number of neighbors
error_rate=[]

for i in range(1,40):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i= knn.predict(X_test)
    error_rate.append(np.mean(pred_i!= y_test))

plt.figure(figsize=(12,8))
plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',\
         markerfacecolor='red', markersize=5)
plt.xlabel('Number of neighbors')
plt.ylabel('error rate')
plt.title('Number of neighbors vs error rate')

# K =23

knn= KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)

pred= knn.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print(confusion_matrix(y_test, pred))

print(accuracy_score(y_test, pred))

# ROC curve
y_pred_proba= knn.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_curve

fpr, tpr, thresholds= roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8,6))
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label= 'KNN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')

from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, y_pred_proba))