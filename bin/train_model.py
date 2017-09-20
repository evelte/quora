import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from quora.aux_functions import get_cols_with_nans, plot_confusion_matrix
from sklearn.externals import joblib

# 7/60 works fine  max_depth=7, max_features='log2', n_estimators=60
# 5/100 pl         max_depth=5, max_features='log2', n_estimators=100
# 3/100 ok         max_depth=5, max_features='log2', n_estimators=200 0.81 no imporve over 100


def train_model(X_train, y_train, store=False):

    # train model
    # ==================================================================================================================
    # split training data into 2 equal parts
    # ------------------------------------------------------------------------------------------------------------------
    # It is important to train the ensemble of trees on a different subset of the training data than the linear
    # regression model to avoid overfitting, in particular if the total number of leaves is similar to the number of
    # training samples

    # split keeping original ratios!
    sss = StratifiedShuffleSplit(n_splits=1, random_state=0, train_size=0.5, test_size=0.5)

    for train_index1, train_index2 in sss.split(X_train, y_train):
        X_train_1, X_train_2 = X_train.iloc[train_index1], X_train.iloc[train_index2]
        y_train_1, y_train_2 = y_train.iloc[train_index1], y_train.iloc[train_index2]

    # train model with optimized parameters
    # ------------------------------------------------------------------------------------------------------------------
    # Supervised transformation based on random forests
    # the number of features is very small, thus max_features is set to None:
    rf = RandomForestClassifier(max_depth=7, max_features=None, n_estimators=100, min_samples_leaf=50)
    rf_enc = OneHotEncoder()
    rf_lm = LogisticRegression()

    rf.fit(X_train_1, y_train_1)
    rf_enc.fit(rf.apply(X_train_1))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_2)), y_train_2)

    return rf, rf_enc, rf_lm


def evaluate_model(X_test, y_test, rf, rf_enc, rf_lm):
    # evaluate model
    # ==================================================================================================================

    # get predictions
    # ------------------------------------------------------------------------------------------------------------------
    y_predict = rf_lm.predict(rf_enc.transform(rf.apply(X_test)))
    y_predict_probabilities = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]

    # if I want to set the prediction on my personalized threshold
    y_predict_customized = [0 if x < 0.4 else 1 for x in y_predict_probabilities]

    # get performance metrics
    # ------------------------------------------------------------------------------------------------------------------
    accuracy = rf_lm.score(rf_enc.transform(rf.apply(X_test)), y_test)
    logloss = log_loss(y_test, y_predict_probabilities)

    # ROC CURVE
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_predict_probabilities)
    roc_auc = auc(fpr_rf_lm, tpr_rf_lm)

    # confusion matrix for test result
    TN, FP, FN, TP = confusion_matrix(y_test, y_predict).ravel()

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
    # ------------------------------------------------------------------------------------------------------------------
    # succeeded: true positives + true negatives
    succeeded = TP+TN

    # failed: false positives + false negatives
    failed = FP+FN

    # print results
    # ------------------------------------------------------------------------------------------------------------------
    print('Accuracy: {}'.format(accuracy))
    print('Logloss: {}'.format(logloss))
    print('Area under curve: {}'.format(roc_auc))

    print(classification_report(y_test.values, y_predict_customized))
    print(classification_report(y_test.values, y_predict))

    print('Total number of tests: {}'.format(TOTAL))
    print('Number of succeeded tests: {}'.format(succeeded))
    print('Number of failed tests: {}'.format(failed))

    # plot confusion matrix
    # ------------------------------------------------------------------------------------------------------------------
    cnf_matrix = confusion_matrix(y_test, y_predict)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['No duplicate', 'duplicate'],
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['No duplicate', 'duplicate'], normalize=True,
                          title='Normalized confusion matrix')

    # plot ROC curve for results
    # ------------------------------------------------------------------------------------------------------------------
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

    return logloss, roc_auc


if __name__ == '__main__':

    # load processed data
    # ==================================================================================================================
    df = pd.read_pickle(os.path.join(quora.root, 'data', 'processing', 'processed_v9_all.pkl'))
    print(df.columns.values)

    # these means and stds have to be stored in the metadata
    # df['word_diff'] = (df.word_diff - df.word_diff.mean()) / df.word_diff.std()
    # df['char_diff'] = (df.char_diff - df.char_diff.mean()) / df.char_diff.std()

    # Set up data to train/test
    # ==================================================================================================================
    X = df[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim', 'cosine_sim2',
            'key_share', 'topic_sim']]
    y = df['is_duplicate']

    # check for missing values
    # ------------------------------------------------------------------------------------------------------------------
    # missing cosine distances... they are filled with the average of the column
    X.fillna(X.mean(), inplace=True)

    # confirm that there are no nans left
    get_cols_with_nans(X)

    # do some hyper-parameter tuning... use complete dataset!
    # ------------------------------------------------------------------------------------------------------------------
    # clf = RandomForestClassifier()
    # tuning_params = dict(max_depth=[None, 5, 7, 10, 15],
    #                      max_features=['sqrt', None, 'log2'],
    #                      min_samples_leaf=[1, 5, 10, 50, 100, 200],
    #                      n_estimators=[25, 50, 75, 100])
    #
    # # define tuning parameters for estimator, optimizing for log loss
    # estimator = GridSearchCV(clf, tuning_params, n_jobs=-1, cv=3, scoring='neg_log_loss')
    #
    # # fit the estimator on the overall training set
    # estimator.fit(X, y)
    # print(estimator.best_params_)

    # divide into train and validation set (80/20)
    # ------------------------------------------------------------------------------------------------------------------
    # normal split into 80/20
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # OPTION 1: train and test for model evaluation
    # ==================================================================================================================
    # split keeping original ratios! use 10 splits for cross validation
    # sss = StratifiedShuffleSplit(n_splits=10, random_state=0, train_size=0.8, test_size=0.2)
    #
    # logloss_overall = []
    # roc_auc_overall = []
    #
    # for train_index, test_index in sss.split(X, y):
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #
    #     # train
    #     rf, rf_enc, rf_lm = train_model(X_train, X_train)
    #     # test
    #     logloss, roc_auc = evaluate_model(X_test, y_test, rf, rf_enc, rf_lm)
    #
    #     logloss_overall.append(logloss)
    #     roc_auc_overall.append(roc_auc)
    #
    # print('Overall performance results:')
    # print('Logloss: {}'.format(np.nanmean(logloss_overall)))
    # print('Area under curve: {}'.format(np.nanmean(roc_auc_overall)))

    # OPTION 2: train final model to use on submission on total dataset
    # ==================================================================================================================
    rf, rf_enc, rf_lm = train_model(X, y)
    logloss, roc_auc = evaluate_model(X, y, rf, rf_enc, rf_lm)

    # store model
    # ------------------------------------------------------------------------------------------------------------------
    joblib.dump(rf, 'model0.pkl')
    joblib.dump(rf_enc, 'model1.pkl')
    joblib.dump(rf_lm, 'model2.pkl')
