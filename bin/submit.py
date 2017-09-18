import os
import quora
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from quora.feature_engineering import extract_features, add_some_columns
from quora.aux_functions import get_cols_with_nans, concatenate_csvs


# load test data
# ======================================================================================================================
file = os.path.join(quora.root, 'data', 'test.csv')
df = pd.DataFrame.from_csv(file, index_col=None)

# there are nans in the test questions! replace them with empty strings
# get_cols_with_nans(df)
df.fillna("", inplace=True)

# process test file in chunks:
# ======================================================================================================================
for chunk, df in df.groupby(np.arange(len(df))//100000):  # chunks of 300,000 would allow ~8 equal partitions

    chunk += 1
    print('Processing chunk #{}'.format(chunk))

    if chunk < 11: continue

    # 1. extract features from test data | test_id, question1, question2
    # ------------------------------------------------------------------------------------------------------------------
    df = add_some_columns(df)
    features = df.apply(lambda row: extract_features(row[1], row[2], row[3], row[4], log=False), axis=1)

    dfs = pd.DataFrame()
    dfs['synsets1'], dfs['synsets2'], dfs['similarity'], dfs['word_share'], dfs['overlap'], dfs['diff'], \
                dfs['word_diff'], dfs['char_diff'], dfs['cosine_sim'], dfs['cosine_sim2'] = zip(*features)

    # 2. define dataframe to apply models on
    # ------------------------------------------------------------------------------------------------------------------
    X = dfs[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim', 'cosine_sim2']]

    # 3. store pre-processed dataframe with test data chunk
    # ------------------------------------------------------------------------------------------------------------------
    joblib.dump(df, os.path.join(quora.root, 'data', 'submission', 'test_X_chunk{}.pkl'.format(chunk)), compress=True)

    # 4. load trained models from file
    # ------------------------------------------------------------------------------------------------------------------
    rf = joblib.load('model0.pkl')
    rf_enc = joblib.load('model1.pkl')
    rf_lm = joblib.load('model2.pkl')

    # 5. apply models to test data chunk
    # ------------------------------------------------------------------------------------------------------------------
    res = rf_lm.predict(rf_enc.transform(rf.apply(X)))

    # 6. construct final dataset | test_id, is_duplicated -- concatenated by columns
    # ------------------------------------------------------------------------------------------------------------------
    df_res = pd.concat([d.reset_index(drop=True) for d in [df.test_id, pd.Series(res)]], axis=1)
    df_res.columns = ['test_id', 'is_duplicated']

    # 7. store result from each chunk to csv
    # ------------------------------------------------------------------------------------------------------------------
    f = os.path.join(quora.root, 'data', 'submission', '{}_submission_chunk{}.csv'.format(str(chunk).zfill(3), chunk))
    df_res.to_csv(f, index=False)


# concatenate results
# ======================================================================================================================
concatenate_csvs()