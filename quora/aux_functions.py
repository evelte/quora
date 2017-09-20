import pandas as pd
import glob, os
import quora
import datetime


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


if __name__ == '__main__':

    # test concatenate csvs
    concatenate_csvs()
