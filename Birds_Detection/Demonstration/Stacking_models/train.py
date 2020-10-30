#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:58:00 2020

@author: marcpozzo
"""

# USAGE
# python train.py

# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.externals import joblib
import pandas as pd

#Load Train and Test sets
train=pd.read_csv('train.csv')
trainX=train.iloc[:,:-1]
trainY=train.iloc[:,-1]

test=pd.read_csv('test.csv')
testX=test.iloc[:,:-1]
testY=test.iloc[:,-1]
### Train with Logistic Regression 


# train the model
print("[INFO] training model...")
model_logit = LogisticRegression(solver="lbfgs", multi_class="auto")
model_logit.fit(trainX, trainY)
model_logit.fit(testX, testY)

# evaluate the model
print("[INFO] evaluating...")
preds = model_logit.predict(testX)
# val proba : model.predict_proba(testX)
print(classification_report(testY, preds))
#print(classification_report(testY, preds, target_names=le.classes_))

# Erreur
model_logit.score(testX, testY)
# Coefficients
model_logit.coef_

# serialize the model to disk
print("[INFO] saving model...")
f = open("model_log", "wb")
f.write(pickle.dumps(model_logit))
f.close()


### Train with DECISIONTREE

# Récupérer le meilleur paramètre pour le model

param=[{"max_depth":list(range(3,16))}]

digit_tree = GridSearchCV(DecisionTreeClassifier(), param, cv=5, n_jobs=-1)

digit_opt=digit_tree.fit(trainX, trainY)


digit_opt.best_params_


# Initialisation de l'arbre de décision 
tree = DecisionTreeClassifier(max_depth = digit_opt.best_params_.get('max_depth'))

digit_tree = tree.fit(trainX, trainY)

1-digit_tree.score(testX,testY)

y_pred = digit_tree.predict(testX) 


#TestY = np.array(TestY, ndmin = 1)

# matrice de confusion
table = pd.crosstab(testY,y_pred)
print(table)

plt.matshow(table)
plt.title("Matrice de Confusion")
plt.colorbar()
plt.show()

# serialize the model to disk
print("[INFO] saving model...")
f = open("output/tree_model", "wb")
f.write(pickle.dumps(digit_tree))
f.close()


### Train with RandomForest

# Récupérer les meilleurs paramètres pour le model

rfc = RandomForestClassifier(random_state=42) 

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,9,10,11,12,13,14,15],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)

CV_rfc.fit(trainX, trainY)

CV_rfc.best_params_


#Initialiser le model avec les paramètres obtenue au-dessus 

rfc1 = RandomForestClassifier(random_state=42, 
                              max_features='auto', 
                              n_estimators= 200, 
                              max_depth=13, 
                              criterion='gini')

# Train the model
RandomForest = rfc1.fit(trainX, trainY)

y_pred = RandomForest.predict(testX)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(testY, y_pred))


# matrice de confusion
table = pd.crosstab(testY,y_pred)
print(table)


# serialize the model to disk
print("[INFO] saving model...")
f = open("output/RandomForest_model", "wb")
f.write(pickle.dumps(RandomForest))
f.close()


### Training with LASSO 

model = Lasso()

alphas = np.array([1,0.1,0.01,0.001,0.0001,0])

grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
#grid = GridSearchCV(Lasso(), {'alpha': [1e-5, 0.01, 0.1, 0.5, 0.8, 1]}, verbose=3)

trainY = np.array(trainY, dtype='float64')
testY = np.array(testY, dtype='float64')

testY = np.array(testY, dtype='float64')

grid.fit(trainX, trainY)

print(grid.best_score_)
print(grid.best_estimator_.alpha)


model = joblib.load("/home/khalid/Bureau/Chaine_de_traitement/output/Lasso_model")
pred = model.predict(testX)
pro = tree.predict_proba(testX)



y_pred = model.predict(testX)

y_pred1 = np.array([],dtype="<U1")
for i in y_pred:
    if(i<0.5):
        y_pred1 = np.append(y_pred1,'0')
    else:
        y_pred1 = np.append(y_pred1,'1')
        
print("Accuracy:",metrics.accuracy_score(testY, y_pred1))

# matrice de confusion
table = pd.crosstab(testY,y_pred1)
print(table)


probas = y_pred

# serialize the model to disk
print("[INFO] saving model...")
f = open("output/Lasso_model", "wb")
f.write(pickle.dumps(grid))
f.close()




