import os
import time
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import quora
from quora.feature_engineering import calc_similarity
from quora.feature_engineering import extract_features, add_some_columns


def process_data():
    """

    :return:
    """

    start_time = time.time()  # count run time

    # load dataframe with word and char counts... remove rows with less than 3 words and drop nans
    # ------------------------------------------------------------------------------------------------------------------
    # TODO: consider some further outlier analysis
    df = pd.read_pickle(os.path.join(quora.root, 'data', 'train_counts.pkl'))
    df = df[(df.q1_n_words > 2) & (df.q2_n_words > 2)]
    df.dropna(inplace=True)

    dfs = df  # run on complete data set
    print(dfs.columns.values)

    # extract features from questions
    # ------------------------------------------------------------------------------------------------------------------
    dfs = add_some_columns(dfs)
    features = dfs.apply(lambda row: extract_features(row[2], row[3], row[9], row[10], row[4]), axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        dfs['synsets1'], dfs['synsets2'], dfs['similarity'], dfs['word_share'], dfs['overlap'], dfs['diff'], \
        dfs['word_diff'], dfs['char_diff'], dfs['cosine_sim'], dfs['cosine_sim2'] = zip(*features)

    # pickle processed dataframe
    dfs.to_pickle(os.path.join(quora.root, 'data', 'processed_v6_all.pkl'))

    elapsed_time = time.time() - start_time  # count run time
    print('Elapsed time: {}'.format(elapsed_time))


if __name__ == '__main__':

    process_data()
