import os
import quora
import pandas as pd
from nltk.corpus import stopwords
import nltk
import string
import re


def replacements(text):

    _original = text

    # normalize text as much as possivle

    # text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r"\be g\b", " eg ", text)
    text = re.sub(r"\bb g\b", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"\b9 11\b", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r"\busa\b", " America ", text)
    text = re.sub(r"\bUSA\b", " America ", text)
    text = re.sub(r"\bu s\b", " America ", text)
    text = re.sub(r"\buk\b", " England ", text)
    text = re.sub(r"\bUK\b", " England ", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r"\bcs\b", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"\bJ K\b", " JK ", text)
    text = re.sub(r"\b1\b", " one ", text)
    text = re.sub(r"\b2\b", " two ", text)
    text = re.sub(r"\b3\b", " three ", text)
    text = re.sub(r"\b4\b", " four ", text)
    text = re.sub(r"\b5\b", " five ", text)
    text = re.sub(r"\b6\b", " six ", text)
    text = re.sub(r"\b7\b", " seven ", text)
    text = re.sub(r"\b8\b", " eight ", text)
    text = re.sub(r"\b9\b", " nine ", text)
    text = re.sub(r"\b0\b", " zero ", text)

    return text


def process(text):

    text = replacements(text)

    punctuation = list(string.punctuation)
    # appending reticências (removed while converting to list)
    punctuation.append('...')
    to_be_removed = set(stopwords.words('english')+ punctuation)

    # without punctuation?
    tokenized = nltk.word_tokenize(text)
    filtered =  [word.lower() for word in tokenized if word not in to_be_removed]

    text = ' '.join(filtered)
    return text


if __name__ == '__main__':

    df = pd.read_pickle(os.path.join(quora.root, 'data', 'train.pkl'))
    df.dropna(inplace=True)

    dfs = df.head(5000)

    for index, row in dfs.iterrows():

        t = row[2]
        t = process(t)
