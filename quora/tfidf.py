from sklearn.feature_extraction.text import TfidfVectorizer
import quora
import pandas as pd
import os
from sklearn.externals import joblib
from quora.pre_processing import process
from quora.feature_engineering import topic_similarity


def train_vectorizer():
    """
    Train TFIDF vectorizer on quora corpus (training data)
    store vectorizer for later use
    :return:
    """

    # load quora *train* corpus
    corpus = pd.read_pickle(os.path.join(quora.root, 'data', 'corpus.pkl'))

    tfidf = TfidfVectorizer()

    # Fit the TfIdf model
    tfidf.fit(corpus)

    # store for later use
    path = os.path.join(quora.root, 'data', 'tfidf2.pkl')
    joblib.dump(tfidf, path)

    print('Dumped vectorizer at {}.'.format(path))


if __name__ == '__main__':

    tfidf = joblib.load(os.path.join(quora.root, 'data', 'tfidf2.pkl'))

    q1 = 'why do girls want to be friends with the guy they reject'
    q2 = 'What is causing someone to be jealous?'

    q1 = process(q1)
    q2 = process(q2)

    topic_sim = topic_similarity(q1, q2)

    print(topic_sim)