import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
import numpy as np
import quora
from sklearn.metrics import roc_curve, auc, log_loss, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
import sys
from quora.aux_functions import get_cols_with_nans
from sklearn.externals import joblib


# load processed data
# ======================================================================================================================
df = pd.read_pickle(os.path.join(quora.root, 'data', 'processed_v9_all.pkl'))
print(df.columns.values)

# these means and stds have to be stored in the metadata
# df['word_diff'] = (df.word_diff - df.word_diff.mean()) / df.word_diff.std()
# df['char_diff'] = (df.char_diff - df.char_diff.mean()) / df.char_diff.std()

# Set up data to train/test
# ======================================================================================================================
X = df[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim', 'cosine_sim2',
        'key_share', 'topic_sim']]
y = df['is_duplicate']

# check for missing values
# ----------------------------------------------------------------------------------------------------------------------

# missing cosine distances... they are filled with the average of the column
X.fillna(X.mean(), inplace=True)

# confirm that there are no nans left
get_cols_with_nans(X)

# divide into train and validation set (80/20)
# ----------------------------------------------------------------------------------------------------------------------
# normal split into 80/20
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# split keeping original ratios!
sss = StratifiedShuffleSplit(n_splits=1, random_state=0, train_size=0.8)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# train model
# ======================================================================================================================

# split training data into 2 equal parts
# ----------------------------------------------------------------------------------------------------------------------
# It is important to train the ensemble of trees on a different subset of the training data than the linear regression
# model to avoid overfitting, in particular if the total number of leaves is similar to the number of training samples

# split keeping original ratios!
sss = StratifiedShuffleSplit(n_splits=1, random_state=0, train_size=0.8)

for train_index1, train_index2 in sss.split(X_train, y_train):
    X_train_1, X_train_2 = X_train.iloc[train_index1], X_train.iloc[train_index2]
    y_train_1, y_train_2 = y_train.iloc[train_index1], y_train.iloc[train_index2]

# to some hyper-parameter tuning...
# ----------------------------------------------------------------------------------------------------------------------
clf = RandomForestClassifier()
tuning_params = dict(max_depth=[None, 5, 7, 10, 15],
                     max_features=['sqrt', None, 'log2'],
                     min_samples_leaf=[1, 5, 10, 50, 100, 200],
                     n_estimators=[25, 50, 75, 100])

# define tuning parameters for estimator, optimizing for log loss
estimator = GridSearchCV(clf, tuning_params, n_jobs=-1, cv=10, scoring='neg_log_loss')

# fit the estimator on the overall training set
estimator.fit(X_train, y_train)

print(estimator.best_params_)
input()

# {'n_estimators': 20, 'max_depth': 7}
# {'n_estimators': 25, 'max_depth': 11}
#
# {'max_depth': 16, 'max_features': 'log2', 'min_samples_leaf': 50}

#
# using log loss as scorer: {'n_estimators': 25, 'max_depth': 13}
# {'max_features': 'log2', 'min_samples_leaf': 10, 'n_estimators': 100}

# 50 estimators, max_depth = 7
# accuracy: 0.7198148927219183
# logloss: 0.49331027603933864
# Area under curve: 0.8047804001460385

# 100 estimators, max_depth = 7
# accuracy: 0.7155584152045337
# logloss: 0.49999864539260774
# Area under curve: 0.7997621537508702
# {'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 100}

# {'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 10, 'n_estimators': 100}


# 7/60 works fine  max_depth=7, max_features='log2', n_estimators=60
# 5/100 pl         max_depth=5, max_features='log2', n_estimators=100
# 3/100 ok         max_depth=5, max_features='log2', n_estimators=200 0.81 no imporve over 100
#            max_depth=5, max_features=None, n_estimators=200 ~same 0.8auc     run2: 0.79
# max_depth=5, max_features=None, n_estimators=500 auc0.79

# estimator = GridSearchCV(cv=None,
#              estimator=LogisticRegression(C=1.0, intercept_scaling=1,
#              dual=False, fit_intercept=True, penalty='l2', tol=0.0001),
#              param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
#
# estimator.fit(X_train, y_train)
# print(estimator.best_params_)
# input()

# y_pred = clf.predict_proba(X_test_o)[:, 1]
# fpr, tpr, _ = roc_curve(y_test_o, y_pred)
# log_loss = log_loss(y_test_o, y_pred)
# roc_auc = auc(fpr, tpr)
# accuracy = clf.score(X_test_o, y_test_o)
# print('logloss: {}'.format(log_loss))
# print('Area under curve: {}'.format(roc_auc))
# print('accuracy: {}'.format(accuracy))

# train model with optimized parameters
# ----------------------------------------------------------------------------------------------------------------------
# Supervised transformation based on random forests
# the number of features is very small, thus max_features is set to None:
rf = RandomForestClassifier(max_depth=7, max_features=None, n_estimators=100, min_samples_leaf=50)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()

rf.fit(X_train_1, y_train_1)
rf_enc.fit(rf.apply(X_train_1))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_2)), y_train_2)

# store model
# ======================================================================================================================
joblib.dump(rf, 'model0.pkl')
joblib.dump(rf_enc, 'model1.pkl')
joblib.dump(rf_lm, 'model2.pkl')

# evaluate model
# ======================================================================================================================

# get predictions
# ----------------------------------------------------------------------------------------------------------------------
y_predict = rf_lm.predict(rf_enc.transform(rf.apply(X_test)))
y_predict_probabilities = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]

# if I want to set the prediction on my personalized threshold
y_predict_customized = [0 if x < 0.4 else 1 for x in y_predict_probabilities]

# get performance metrics
# ----------------------------------------------------------------------------------------------------------------------
accuracy = rf_lm.score(rf_enc.transform(rf.apply(X_test)), y_test)
log_loss = log_loss(y_test, y_predict_probabilities)

# ROC CURVE
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_predict_probabilities)
roc_auc = auc(fpr_rf_lm, tpr_rf_lm)

# res_coef = pd.DataFrame(zip(X_train_lr.columns, np.transpose(model.coef_)))
print(X_train.columns.values)
print(rf_lm.coef_)
print(len(rf_lm.coef_[0]))
print(len(rf_lm.coef_))

# confusion matrix for test result
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)

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
TOTAL = TP+FP+FN+TN
ACC = (TP+TN)/TOTAL

# number of succeeded/failed tests
# ----------------------------------------------------------------------------------------------------------------------
# succeeded: true positives + true negatives / all tests (ACCURACY)
succeeded = (TP+TN)/(TP+FP+FN+TN)

# failed: false positives + false negatives / all tests
failed = (FP+FN)/(TP+FP+FN+TN)

# print results
# ----------------------------------------------------------------------------------------------------------------------
print('Accuracy: {}'.format(accuracy))
print('Logloss: {}'.format(log_loss))
print('Area under curve: {}'.format(roc_auc))
print('ACC: {}'.format(ACC))

print(classification_report(y_test.values, y_predict_customized))
print(classification_report(y_test.values, y_predict))

print('Total number of tests: {}'.format(TOTAL))
print('Number of succeeded tests: {}'.format(succeeded))
print('Number of failed tests: {}'.format(failed))

# plot ROC curve for results
# ----------------------------------------------------------------------------------------------------------------------
plt.figure()
lw = 2
plt.plot(fpr_rf_lm, tpr_rf_lm, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
