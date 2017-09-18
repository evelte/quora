import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import quora


# load processed data
df = pd.read_pickle(os.path.join(quora.root, 'data', 'processed_v6_all.pkl'))

# drop nans from dataframe
df.dropna(inplace=True)

# path to save plots
path = os.path.join(quora.root, 'data', 'plots')

# 1. take a look at the word and char counts
all_words = pd.concat([df.q1_n_words, df.q2_n_words], axis=0)
all_chars = pd.concat([df.q1_n_chars, df.q2_n_chars], axis=0)

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=False)
sns.boxplot(all_words, color='green', ax=ax1)
sns.boxplot(all_chars, color='green', ax=ax2)
fig.suptitle('Boxplot for word and char count')
ax1.set_xlabel('Word count')
ax2.set_xlabel('Character count')
fig.savefig(os.path.join(path, "word_and_char_count.png"))

# cols = ['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim']
#
# df = pd.read_pickle(os.path.join(quora.root, 'data', 'processed_v3_5000.pkl'))
#
# # conditional coloring based one one variable (should be categorical)
# sns.pairplot(df[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim', 'is_duplicate']],
#              hue='is_duplicate')
#
# sns.plt.show()
#
# for col in cols:
#     sns.plt.figure()
#     # sns.distplot(df[df['is_duplicate'] == 1.0][col], color='red')
#     # sns.distplot(df[df['is_duplicate'] == 0.0][col], color='green')
#
#     sns.violinplot(x='is_duplicate', y=col, data=df)
#
# sns.plt.show()
