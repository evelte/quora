import os
import quora
import pandas as pd
from quora.feature_engineering import extract_features
import shelve
from sklearn.externals import joblib


# load test data
file = os.path.join(quora.root, 'data', 'test.csv')
df = pd.DataFrame.from_csv(file, index_col=None)

print(df.shape)
print(df.columns.values)

df = df[0:10]
ids = df['test_id']

# extract features from test data
# test_id, question1, question2
features = df.apply(lambda row: extract_features(row[1], row[2]), axis=1)

dfs = pd.DataFrame()

dfs['synsets1'], dfs['synsets2'], dfs['similarity'], dfs['word_share'], dfs['overlap'], dfs['diff'], \
            dfs['word_diff'], dfs['char_diff'] = zip(*features)

X = dfs[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff']]

# # load from file
# model = joblib.load('model.pkl')
#
# # apply trained model
# res = pd.Series(model.predict(X))

# # load from file
rf = joblib.load('model0.pkl')
rf_enc = joblib.load('model1.pkl')
rf_lm = joblib.load('model2.pkl')

res = pd.Series(rf_lm.predict(rf_enc.transform(rf.apply(X))))


# store result to csv
# test_id, is_duplicated
df_res = pd.concat([ids, res], axis=1, ignore_index=True)
df_res.columns = ['test_id', 'is_duplicated']
print(df_res)

df_res.to_csv('submission_v2.csv', index=False)
