import nltk, string
import warnings
import spacy
import os
import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import quora
from quora.pre_processing import process
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


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
    """
    normalized difference in length
    :param q1:
    :param q2:
    :return:
    """

    n_words1 = len(q1.split(" "))
    n_words2 = len(q2.split(" "))

    words_len_diff = abs(n_words1 - n_words2)
    char_len_diff = abs(len(q1) - len(q2))

    max_words = max(n_words1, n_words2)
    if max_words != 0:
        words_len_diff /= max_words

    max_chars = max(len(q1), len(q2))
    if max_chars != 0:
        char_len_diff /= max_chars

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
tfidf = joblib.load(os.path.join(quora.root, 'data', 'tfidf2.pkl'))


def cosine_sim(q1, q2):

    try:
        tfidf = vectorizer.fit_transform([q1, q2])
        cosine_similarity = (tfidf * tfidf.T).A[0, 1]
    except:
        # ValueError: ('empty vocabulary; perhaps the documents only contain stop words', 'occurred at index 9581')
        return 0
    else:
        return cosine_similarity


def get_question_key(question):
    """
    these are the key question terms: Who, What, Why, When, Where, How, How Much
    additionally: None and Mixed
    :param q1:
    :param q2:
    :return:
    """

    keys = ['who', 'what', 'why', 'when', 'where', 'how', 'which']
    words = nltk.word_tokenize(question.lower())

    intersect = set(words).intersection(keys)
    if len(intersect) == 0:
        key_word = 'None'
    elif len(intersect) > 1:
        key_word = 'Mixed'
    else:
        key_word = list(intersect)[0]

    return key_word


def get_question_key_share(key1, key2):

    if 'None' in [key1, key2] or 'Mixed' in [key1, key2]:
        key_share = False
    elif key1 == key2:
        key_share = True
    else:
        key_share = False

    return key_share


def topic_similarity(q1, q2):

    def get_top_topic(q):
        X = tfidf.transform([q])
        words = nltk.word_tokenize(q.lower())

        t = []
        for word in words:
            try:
                t.append(X[0, tfidf.vocabulary_[word]])
            except:
                pass

        # t = [X[0, tfidf.vocabulary_[word]] for word in words]
        l = len(t) if len(t) < 3 else 3
        inds = np.argsort(t)[-l:]

        res = []
        for ind in inds:
            try:
                res.append(wn.synsets(words[ind])[0].lemma_names()[0])
            except:
                pass

        # [wn.synsets(words[ind])[0].lemma_names()[0] for ind in inds]

        return res, l

    try:
        top_topic1, l1 = get_top_topic(q1)
        top_topic2, l2 = get_top_topic(q2)

        overlap = len(set(top_topic1).intersection(top_topic2))/max(l1, l2)

        # dic = {wn.synset(s1[idx]).lemma_names()[0]: X[0, tfidf.vocabulary_[word]]
        #        for idx, word in enumerate(nltk.word_tokenize(q1.lower()))}
        #
        # for key, value in dic.items():
        #     print(key, value)
    except:
        overlap = 0.5

    return overlap

def penn_to_wn(tag):
    """
    Convert between a Penn Treebank tag to a simplified Wordnet tag

    :param tag:
    :return:
    """

    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    """
    compute the sentence similarity using Wordnet
    :param sentence1:
    :param sentence2:
    :return:
    """

    # Tokenize and tag

    # part of speech tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        scores = [synset.path_similarity(ss) for ss in synsets2 if synset.path_similarity(ss)]
        best_score = max(scores) if scores else None

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    try:
        # Average the values
        score /= count
    except ZeroDivisionError:
        score = 0
    except:
        raise

    return score, [x.name() for x in synsets1], [x.name() for x in synsets2]


def symmetric_sentence_similarity(sentence1, sentence2):
    """
    compute the symmetric sentence similarity using Wordnet

    :param sentence1:
    :param sentence2:
    :return:
    """

    score1, s1, s2 = sentence_similarity(sentence1, sentence2)
    score2, s2, s1 = sentence_similarity(sentence2, sentence1)

    return (score1+score2) / 2, s1, s2


def extract_features(q1, q2, vec1, vec2, result=None, log=True):

    # TODO: investigate things like semantic similarity and word order similarity

    global processed_rows

    # q1 = 'Elena buys new car china wants sell now in england. what is the best way to transport car.'
    # q2 = 'where to buy red big apples.'

    # get key from raw questions
    q1_key = get_question_key(q1)
    q2_key = get_question_key(q2)

    key_share = get_question_key_share(q1_key, q2_key)

    q1 = process(q1)
    q2 = process(q2)

    s1, s2, similarity = calc_similarity(q1, q2)  # synsets and similarity
    word_share = normalized_word_share(q1, q2)  # common words
    words_len_diff, char_len_diff = len_diff(q1, q2)  # words/chars length difference
    overlap, diff = normalized_synset_share(s1, s2)  # intersection and difference of synsets
    c_similarity = cosine_sim(q1, q2)
    c_similarity2 = 1 - spatial.distance.cosine(vec1, vec2)
    topic_sim = topic_similarity(q1, q2)

    # we don't want nans here...
    if np.isnan(c_similarity2): c_similarity2 = 0.5

    # status update of total number of rows processed
    processed_rows += 1
    print('Processed rows: {}'.format(processed_rows))

    if log:
        print('Q1: {}'.format(q1))
        print('Q2: {}'.format(q2))
        print('Key share: {}'.format(key_share))
        print('Similarity: {}'.format(similarity))
        print('Overlap: {}'.format(overlap))
        print('Diff: {}'.format(diff))
        print('Cosine sim1: {}'.format(c_similarity))
        print('Cosine sim2: {}'.format(c_similarity2))
        print('Topic sim: {}'.format(topic_sim))
        print('Result: {}'.format('Duplicated' if result == 1 else 'No duplicated'))
        print('\n')

    return s1, s2, similarity, word_share, overlap, diff, words_len_diff, char_len_diff, c_similarity, c_similarity2, \
           key_share, topic_sim
