import pandas as pd
import glob, os
import quora
import datetime
import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_cols_with_nans(df):
    """
    identify columns in dataframe with missing values
    :param df:
    :return:
    """

    for col in df.columns.values:
        if df[col].isnull().any():
            print(col, df[col].dtype, 'missing')
        else:
            print(col, df[col].dtype, 'no missing values')


def concatenate_csvs(folder):
    """
    concatenate submission files into one
    loads all csv files from data/submission and concatenates them into one big csv file (saved with timestamp)
    :return:
    """

    try:
        path = os.path.join(quora.root, 'data', 'submission', folder)
        df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv"))))

        # make sure the columns have the correct name
        df.columns = ['test_id', 'is_duplicate']

        # make sure the resolts are sorted by test_id
        result = df.sort_values(by='test_id')

        csv_file = os.path.join(quora.root, 'data', 'submission_{}.csv'.format(datetime.datetime.now()))
        result.to_csv(csv_file, ',', index=False)

    except Exception as err:
        print(err)
        print("Error while concatenating submission csv's.")
    else:
        print("Submission csv's concatenated with success into: {}".format(csv_file))


# FROM SCIKIT LEARN
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':

    # test concatenate csvs
    concatenate_csvs()
