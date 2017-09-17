import pandas as pd
import os
import quora


# corpus = pd.read_pickle(os.path.join(quora.root, 'data', 'corpus.pkl'))
#
df = pd.read_pickle(os.path.join(quora.root, 'data', 'train_counts.pkl'))

# remove rows with less than 3 words
df = df[(df.q1_n_words > 2) & (df.q2_n_words > 2)]

df.dropna(inplace=True)

# run on complete dataset
df = df.head(10)

# exctract word2vec vectors
import spacy
import numpy as np

nlp = spacy.load('en')
print('loaded spacy')

vecs1 = [doc.vector for doc in nlp.pipe(df['question1'], n_threads=50)]
vecs1 = np.array(vecs1)
df['q1_feats'] = list(vecs1)

for vec in list(vecs1):
    print(vec)

vecs2 = [doc.vector for doc in nlp.pipe(df['question2'], n_threads=50)]
vecs2 = np.array(vecs2)
df['q2_feats'] = list(vecs2)

print(df)

# save features
pd.to_pickle(df, os.path.join(quora.root, 'data', 'dt_processed.pkl'))

df = pd.read_pickle(os.path.join(quora.root, 'data', 'dt_processed.pkl'))

print(df)

from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity


for index, row in df.iterrows():

    print(row[2])
    print(row[3])

    print(1 - spatial.distance.cosine(row[9], row[10]))
    print(cosine_similarity(row[9], row[10]))
    input()