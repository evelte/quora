import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
import numpy as np
import quora
from sklearn.metrics import roc_curve, auc, log_loss, classification_report
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns
import sys


df = pd.read_pickle(os.path.join(quora.root, 'data', 'processed_v6_5000.pkl'))
print(df.columns.values)

# these means and stds have to be stored in the metadata
# df['word_diff'] = (df.word_diff - df.word_diff.mean()) / df.word_diff.std()
# df['char_diff'] = (df.char_diff - df.char_diff.mean()) / df.char_diff.std()

# X = df[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim']]
# X = df[['similarity', 'word_share', 'overlap', 'diff', 'cosine_sim']]
X = df[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim', 'cosine_sim2']]
y = df['is_duplicate']


# divide into train and validation set (80/20)
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# keep original ratios!
sss = StratifiedShuffleSplit(n_splits=1, random_state=0, train_size=0.8)

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

# clf = RandomForestClassifier()
# tuning_params = dict(max_depth=list(range(1,20)), n_estimators=[5, 10, 15, 20, 25])
# estimator = GridSearchCV(clf, tuning_params)
# estimator.fit(X_train, y_train)
#
# print(estimator.best_params_)
# # {'n_estimators': 20, 'max_depth': 7}
# # {'n_estimators': 25, 'max_depth': 11}
#
# input()


# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=7, n_estimators=20)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()

rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

# res_coef = pd.DataFrame(zip(X_train_lr.columns, np.transpose(model.coef_)))
print(X_train.columns.values)
print(rf_lm.coef_)
print(len(rf_lm.coef_[0]))
print(len(rf_lm.coef_))

accuracy = rf_lm.score(rf_enc.transform(rf.apply(X_test)), y_test)
print('accuracy: {}'.format(accuracy))

y_predict = rf_lm.predict(rf_enc.transform(rf.apply(X_test)))
y_prediction = [0 if x <0.4 else 1 for x in y_pred_rf_lm]

log_loss = log_loss(y_test, y_pred_rf_lm)
roc_auc = auc(fpr_rf_lm, tpr_rf_lm)

print('logloss: {}'.format(log_loss))
print('Area under curve: {}'.format(roc_auc))

print(classification_report(y_test.values, y_prediction))
print(classification_report(y_test.values, y_predict))

# scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
# print(scores.mean())

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

from sklearn.externals import joblib

joblib.dump(rf, 'model0.pkl')
joblib.dump(rf_enc, 'model1.pkl')
joblib.dump(rf_lm, 'model2.pkl')

sys.exit()
# ======================================================================================================================
