import nltk, string
import warnings
import spacy
import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from quora.wordnet_test import symmetric_sentence_similarity
from quora.pre_processing import process


nlp = spacy.load('en')
processed_rows = 0
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def add_some_columns(df):
    """
    :param df:
    :return: dataframe with 2 new columns: q1_feats and q2_feats
    """

    print('adding columns with vectorized questions')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # there are some nans in the test data...
        vecs1 = [doc.vector for doc in nlp.pipe(df['question1'], n_threads=50)]
        vecs1 = list(np.array(vecs1))
        df['q1_feats'] = vecs1

        vecs2 = [doc.vector for doc in nlp.pipe(df['question2'], n_threads=50)]
        vecs2 = list(np.array(vecs2))
        df['q2_feats'] = vecs2

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

    try:
        tfidf = vectorizer.fit_transform([q1, q2])
        cosine_similarity = (tfidf * tfidf.T).A[0, 1]
    except:
        # ValueError: ('empty vocabulary; perhaps the documents only contain stop words', 'occurred at index 9581')
        return 0
    else:
        return cosine_similarity


def extract_features(q1, q2, vec1, vec2, result=None, log=True):

    global processed_rows

    q1 = process(q1)
    q2 = process(q2)

    s1, s2, similarity = calc_similarity(q1, q2)  # synsets and similarity
    word_share = normalized_word_share(q1, q2)  # common words
    words_len_diff, char_len_diff = len_diff(q1, q2)  # words/chars length difference
    overlap, diff = normalized_synset_share(s1, s2)  # intersection and difference of synsets
    c_similarity = cosine_sim(q1, q2)
    c_similarity2 = 1 - spatial.distance.cosine(vec1, vec2)

    # we don't want nans here...
    if np.isnan(c_similarity2): c_similarity2 = 0.5

    # status update of total number of rows processed
    processed_rows += 1
    print('Processed rows: {}'.format(processed_rows))

    if log:
        print('Q1: {}'.format(q1))
        print('Q2: {}'.format(q2))
        print('Similarity: {}'.format(similarity))
        print('Overlap: {}'.format(overlap))
        print('Diff: {}'.format(diff))
        print('Cosine sim1: {}'.format(c_similarity))
        print('Cosine sim2: {}'.format(c_similarity2))
        print('Result: {}'.format('Duplicated' if result == 1 else 'No duplicated'))
        print('\n')

    return s1, s2, similarity, word_share, overlap, diff, words_len_diff, char_len_diff, c_similarity, c_similarity2
