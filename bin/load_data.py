import pandas as pd
import quora
import os
import warnings
import numpy as np
from quora.feature_engineering import calc_similarity
from quora.feature_engineering import extract_features, add_some_columns
from sklearn.model_selection import train_test_split


def load_data(name):
    """
    load csv data into dataframe and pickle for quick access
    two dataframes are stored to disk:
    1) original data, pickled
    2) original data with 2 new columns: number of word and number of char count for each question in the pair
    :return:
    """

    file = os.path.join(quora.root, 'data', '{}.csv'.format(name))
    df = pd.DataFrame.from_csv(file)

    df.to_pickle(os.path.join(quora.root, 'data', '{}.pkl'.format(name)))

    # add columns for word and char counts
    df['q1_n_words'] = df['question1'].apply(lambda x: len(x.split(" ")) if isinstance(x, str) else np.nan)
    df['q1_n_chars'] = df['question1'].apply(lambda x: len(x) if isinstance(x, str) else np.nan)

    df['q2_n_words'] = df['question2'].apply(lambda x: len(x.split(" ")) if isinstance(x, str) else np.nan)
    df['q2_n_chars'] = df['question2'].apply(lambda x: len(x) if isinstance(x, str) else np.nan)

    df.to_pickle(os.path.join(quora.root, 'data', '{}_counts.pkl'.format(name)))


def store_all_train_questions():
    """
    create and store data series with all training questions (removing nans)
    :return:
    """

    # load original dataset from file
    df = pd.read_pickle(os.path.join(quora.root, 'data', 'train.pkl'))

    # concatenate questions into one data series
    corpus = pd.concat([df.question1, df.question2], axis=0)

    # drop rows with missing question (nan)
    corpus.dropna(inplace=True)

    # store series to file for easy access
    corpus.to_pickle(os.path.join(quora.root, 'data', 'corpus.pkl'))


if __name__ == '__main__':

    # first time load and process data
    # load_data('train')
    # store_all_train_questions()

    import time
    start_time = time.time()

    df = pd.read_pickle(os.path.join(quora.root, 'data', 'train_counts.pkl'))

    # remove rows with less than 3 words
    df = df[(df.q1_n_words > 2) & (df.q2_n_words > 2)]

    df.dropna(inplace=True)

    # run on complete dataset
    dfs = df
    # dfs = df.head(5000)
    print(dfs.columns.values)

    dfs = add_some_columns(dfs)
    features = dfs.apply(lambda row: extract_features(row[2], row[3], row[9], row[10], row[4]), axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        dfs['synsets1'], dfs['synsets2'], dfs['similarity'], dfs['word_share'], dfs['overlap'], dfs['diff'], \
            dfs['word_diff'], dfs['char_diff'], dfs['cosine_sim'], dfs['cosine_sim2'] = zip(*features)

    print(dfs)

    dfs.to_pickle(os.path.join(quora.root, 'data', 'processed_v6_all.pkl'))

    elapsed_time = time.time() - start_time
    print('Elapsed time: {}'.format(elapsed_time))
