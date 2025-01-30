#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
reg = LogisticRegression()
warnings.filterwarnings('ignore')

# data = '../dataset/fbs_nas.csv'
# data = '../dataset/fbs_rrc.csv'
data = '../dataset/msa_nas.csv'
# data = '../dataset/msa_rrc.csv'

df = pd.read_csv(data)
# df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.2'])
# df = df.drop(columns=['Unnamed: 0'])
# df.to_csv('dataset/cleaned_data_new.csv', index=False)

X = df.drop(['label'], axis=1)
y = df['label']

def non_shuffling_train_test_split(X, y, split_at, test_size=0.2):
    i = split_at
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X, y, split_at=1561, test_size = 0.33)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf_list = ['rf', 'svm', 'dt', 'xgb', 'knn', 'nb', 'lr']

def get_model(clf):
    if clf == 'rf':
        model = RandomForestClassifier(criterion='gini', max_depth=3, random_state=0)
    elif clf == 'svm':
        model = SVC()
    elif clf == 'dt':
        model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
    elif clf == 'xgb':
        model = xgb.XGBClassifier(random_state=42)
    elif clf == 'dt':
        model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
    elif clf == 'knn':
        model = KNeighborsClassifier()
    elif clf == 'nb':
        model = GaussianNB()
    elif clf == 'lr':
        model = LogisticRegression()
    return model

for clf in clf_list:
    model = get_model(clf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(clf, '\t{0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# model = get_model('xgb')
model = get_model('knn')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(clf, '\t{0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)
print(classification_report(y_test, y_pred))

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)

TP, FP, TN, FN = perf_measure(list(y_test), list(y_pred))

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print("TPR = ", TPR, "\nFPR = ",FPR, "\nTNR = ", TNR, "\nFNR = ", FNR, "\nFDR = ", FDR, "\nACC = ", ACC)


