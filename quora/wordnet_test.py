from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


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
