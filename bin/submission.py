import os
import quora
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from quora.feature_engineering import extract_features, add_some_columns
from quora.aux_functions import get_cols_with_nans, concatenate_csvs


# 1. load test data
# ======================================================================================================================
file = os.path.join(quora.root, 'data', 'test.csv')
df = pd.DataFrame.from_csv(file, index_col=None)

# there are nans in the test questions! replace them with empty strings
# get_cols_with_nans(df)
df.fillna("", inplace=True)

# 2. process/load test file in chunks:
# ======================================================================================================================
for chunk, df in df.groupby(np.arange(len(df)) // 100000):  # chunks of 300,000 would allow ~8 equal partitions

    chunk += 1

    option = 'load'  # CHANGE OPTION HERE! options: load or process
    load_v = 'v3'
    store_v = 'v3'
    model_v = 'v3'

    if option == 'process':  # do not change this

        print('Processing chunk #{}'.format(chunk))

        # 1. extract features from test data | test_id, question1, question2
        # --------------------------------------------------------------------------------------------------------------
        df = add_some_columns(df)
        features = df.apply(lambda row: extract_features(row[1], row[2], row[3], row[4], log=False), axis=1)

        dfs = pd.DataFrame()
        dfs['synsets1'], dfs['synsets2'], dfs['similarity'], dfs['word_share'], dfs['overlap'], dfs['diff'], \
        dfs['word_diff'], dfs['char_diff'], dfs['cosine_sim'], dfs['cosine_sim2'], dfs['key_share'], dfs['topic_sim'] \
            = zip(*features)

        # 3. store pre-processed dataframe with test data chunk
        # --------------------------------------------------------------------------------------------------------------
        joblib.dump(dfs, os.path.join(quora.root, 'data', 'submission', store_v,
                    'test_X_chunk{}.pkl'.format(chunk)),
                    compress=True)

        ids = df.test_id

    elif option == 'load':  # do not change this

        print('Loading chunk #{}'.format(chunk))

        # specify version here...
        dfs = joblib.load(os.path.join(quora.root, 'data', 'submission', load_v, 'test_X_chunk{}.pkl'.format(chunk)))

        interval = ((chunk - 1) * 100000)
        start = 0 + interval
        end = start + 100000
        ids = pd.Series(range(start, end))

        # the last chunk will have more ids than data points
        if len(dfs) < len(ids):
            ids = ids[0:len(dfs)]

    # 3. define dataframe to apply models on
    # ------------------------------------------------------------------------------------------------------------------
    X = dfs[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim', 'cosine_sim2',
             'key_share', 'topic_sim']]

    # 4. load trained models from file
    # ------------------------------------------------------------------------------------------------------------------
    path = os.path.join(quora.root, 'data', 'models', model_v)

    rf = joblib.load(os.path.join(path, 'model0.pkl'))
    rf_enc = joblib.load(os.path.join(path, 'model1.pkl'))
    rf_lm = joblib.load(os.path.join(path, 'model2.pkl'))

    # 5. apply models to test data chunk
    # ------------------------------------------------------------------------------------------------------------------
    res = rf_lm.predict_proba(rf_enc.transform(rf.apply(X)))[:, 1]

    # 6. construct final dataset | test_id, is_duplicated -- concatenated by columns
    # ------------------------------------------------------------------------------------------------------------------
    df_res = pd.concat([d.reset_index(drop=True) for d in [ids, pd.Series(res)]], axis=1)
    df_res.columns = ['test_id', 'is_duplicated']

    # 7. store result from each chunk to csv
    # ------------------------------------------------------------------------------------------------------------------
    file = '{}_submission_chunk{}.csv'.format(str(chunk).zfill(3), chunk)
    path = os.path.join(quora.root, 'data', 'submission', store_v, file)
    df_res.to_csv(path, index=False)

# concatenate results
# ======================================================================================================================
concatenate_csvs(store_v)
