#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:28:18 2019

@author: bixi
"""

from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

# use the digits given by sklearn
digits = load_digits()
# divide the digits into train set and test set, each set has coresponding label set
X_train,X_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=1)

diction = {0:'KNeighborsClassifier'}

# initialize the KNeighborsClassifier model
knns = KNeighborsClassifier(n_neighbors=3, algorithm='auto', weights='distance', n_jobs=1)
knns.fit(X_train,y_train)
knn_predicts = knns.predict(X_test)
knn_accuracy = metrics.accuracy_score(y_test,knn_predicts)
knn_recall = metrics.recall_score(y_test,knn_predicts,average = 'macro')
knn_matrix = metrics.confusion_matrix(y_test,knn_predicts,labels = digits.target_names)

# initialize the SGDClassifier model
sgd = SGDClassifier()
sgd.fit(X_train,y_train)
sgd_predicts = sgd.predict(X_test)
sgd_accuracy = metrics.accuracy_score(y_test,sgd_predicts)
sgd_recall = metrics.recall_score(y_test,sgd_predicts,average = 'macro')
sgd_matrix = metrics.confusion_matrix(y_test,sgd_predicts,labels = digits.target_names)

# initialize the DecisionTreeClassifer model
dtr = DecisionTreeClassifier()
dtr.fit(X_train,y_train)
dtr_predicts = dtr.predict(X_test)
dtr_accuracy = metrics.accuracy_score(y_test,dtr_predicts)
dtr_recall = metrics.recall_score(y_test,dtr_predicts,average = 'macro')
dtr_matrix = metrics.confusion_matrix(y_test,dtr_predicts,labels = digits.target_names)

print('Results Show',end='\n\n')
print('Test size = 0.25')
print('KNN Accuracy:  %.3f'%knn_accuracy,end='    ')
print('Recall:  %.3f'%knn_recall)
print('SGD Accuracy:  %.3f'%sgd_accuracy,end='    ')
print('Recall:  %.3f'%sgd_recall)
print('DT  Accuracy:  %.3f'%dtr_accuracy,end='    ')
print('Recall:  %.3f'%dtr_recall)
print()
print('KNN Confusion Matrix:')
# print the best confusion matrix
print(knn_matrix)
