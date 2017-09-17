import pandas as pd
import quora
import numpy as np
import warnings
from quora.wordnet_test import symmetric_sentence_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk, string
from quora.pre_processing import process
import spacy
from scipy import spatial


nlp = spacy.load('en')
processed_rows = 0
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def add_some_columns(df):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        vecs1 = [doc.vector for doc in nlp.pipe(df['question1'], n_threads=50)]
        vecs1 = np.array(vecs1)
        df['q1_feats'] = list(vecs1)

        vecs2 = [doc.vector for doc in nlp.pipe(df['question2'], n_threads=50)]
        vecs2 = np.array(vecs2)
        df['q2_feats'] = list(vecs2)

    return df


def calc_similarity(q1, q2):
    """
    :param row:
    :return:
    """

    try:
        similarity, s1, s2 = symmetric_sentence_similarity(q1, q2)
    except Exception as err:
        print(err)
        return np.nan, np.nan, np.nan
    else:
        return s1, s2, similarity


def normalized_word_share(q1, q2):

    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))

    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))


def len_diff(q1, q2):

    n_words1 = len(q1.split(" "))
    n_words2 = len(q2.split(" "))

    words_len_diff = abs(n_words1 - n_words2)
    char_len_diff = abs(len(q1) - len(q2))

    return words_len_diff, char_len_diff


def normalized_synset_share(syns1, syns2):

    overlap = len(set(syns1).intersection(syns2))
    diff = len(list(set(syns1) - set(syns2)))

    ma = max([len(set(syns1)), len(set(syns2))])

    if ma != 0:
        overlap /= ma
        diff /= ma

    return overlap, diff


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


def normalize_text(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


vectorizer = TfidfVectorizer(tokenizer=normalize_text, stop_words='english')


def cosine_sim(q1, q2):
    tfidf = vectorizer.fit_transform([q1, q2])
    return (tfidf * tfidf.T).A[0, 1]


def extract_features(q1, q2, vec1, vec2, result):

    global processed_rows

    q1 = process(q1)
    q2 = process(q2)

    # synsets and similarity
    s1, s2, similarity = calc_similarity(q1, q2)

    # common words
    word_share = normalized_word_share(q1, q2)

    # words/chars length difference
    words_len_diff, char_len_diff = len_diff(q1, q2)

    # intersection and difference of synsets
    overlap, diff = normalized_synset_share(s1, s2)

    try:
        c_similarity = cosine_sim(q1, q2)
    except:
        # ValueError: ('empty vocabulary; perhaps the documents only contain stop words', 'occurred at index 9581')
        c_similarity = 0

    c_similarity2 = 1 - spatial.distance.cosine(vec1, vec2)

    result = 'Duplicated' if result ==1 else 'No duplicated'

    processed_rows += 1
    print('Processed rows: {}'.format(processed_rows))
    print('Q1: {}'.format(q1))
    print('Q2: {}'.format(q2))
    print('Similarity: {}'.format(similarity))
    print('Overlap: {}'.format(overlap))
    print('Diff: {}'.format(diff))
    print('Cosine sim1: {}'.format(c_similarity))
    print('Cosine sim2: {}'.format(c_similarity2))
    print('Result: {}'.format(result))
    print('\n')

    return s1, s2, similarity, word_share, overlap, diff, words_len_diff, char_len_diff, c_similarity, c_similarity2
