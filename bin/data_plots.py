import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import quora


# get data and setup
# ======================================================================================================================

# load processed data
df = pd.read_pickle(os.path.join(quora.root, 'data', 'processing', 'processed_v9_all.pkl'))

# drop nans from dataframe
df.dropna(inplace=True)

# path to save plots
path = os.path.join(quora.root, 'data', 'plots')


# plot stuff!
# ======================================================================================================================

# 1. take a look at the word and char counts
# ----------------------------------------------------------------------------------------------------------------------
all_words = pd.concat([df.q1_n_words, df.q2_n_words], axis=0)
all_chars = pd.concat([df.q1_n_chars, df.q2_n_chars], axis=0)

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False)
sns.boxplot(all_words, color='green', ax=ax1)
sns.boxplot(all_chars, color='green', ax=ax2)
fig.suptitle('Boxplot for word and char count')
ax1.set_xlabel('Word count')
ax2.set_xlabel('Character count')
fig.savefig(os.path.join(path, "word_and_char_count.png"))


# 2. pair plot of all extracted features
# ----------------------------------------------------------------------------------------------------------------------
# conditional coloring based on one variable - is_duplicate
fig, ax = plt.subplots()
sns.pairplot(df[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim', 'key_share',
                 'topic_sim', 'is_duplicate']], hue='is_duplicate')
fig.savefig(os.path.join(path, "pair_plot.png"))


# 3. violin plot for extracted features: compare is_duplicate True/False
# ----------------------------------------------------------------------------------------------------------------------
cols = ['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim', 'key_share', 'topic_sim']

for col in cols:
    fig, ax = plt.subplots()
    sns.violinplot(x='is_duplicate', y=col, data=df)
    fig.savefig(os.path.join(path, "violin_{}.png".format(col)))


# 3.b show distribution instead of violin
# ----------------------------------------------------------------------------------------------------------------------
for col in cols:
    fig, ax = plt.subplots()
    sns.distplot(df[df['is_duplicate'] == 1.0][col], color='red')
    sns.distplot(df[df['is_duplicate'] == 0.0][col], color='green')
    fig.savefig(os.path.join(path, "distribution_{}.png".format(col)))
