import pandas as pd
import quora
import numpy as np
import warnings
from quora.wordnet_test import symmetric_sentence_similarity


processed_rows = 0


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

    # df['q1_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
    # df['q2_words'] = df['question2'].apply(lambda row: len(row.split(" ")))
    # df['words_len_diff'] = abs(df['q1_words'] - df['q2_words'])
    #
    # # character len
    # df['q1_chars'] = df['question1'].str.len()
    # df['q2_chars'] = df['question2'].str.len()
    # df['char_len_diff'] = abs(df['q1_chars'] - df['q2_chars'])

    words_len_diff = abs(len(q1.split(" ")) - len(q2.split(" ")))
    char_len_diff = abs(len(q1) - len(q2))

    return words_len_diff, char_len_diff


def normalized_synset_share(syns1, syns2):

    overlap = len(set(syns1).intersection(syns2))
    diff = len(list(set(syns1) - set(syns2)))

    ma = max([len(set(syns1)), len(set(syns2))])
    mi = min([len(set(syns1)), len(set(syns2))])

    if overlap != 0:
        overlap /= ma
    if diff != 0:
        diff /= mi

    return overlap/ma, diff/mi


def extract_features(q1, q2):

    global processed_rows

    # synsets and similarity
    s1, s2, similarity = calc_similarity(q1, q2)

    # common words
    word_share = normalized_word_share(q1, q2)

    # words/chars length difference
    words_len_diff, char_len_diff = len_diff(q1, q2)

    # intersection and difference of synsets
    overlap, diff = normalized_synset_share(s1, s2)

    processed_rows += 1
    print(processed_rows)

    return s1, s2, similarity, word_share, overlap, diff, words_len_diff, char_len_diff