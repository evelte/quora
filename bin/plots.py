import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import quora


# df = pd.read_pickle(os.path.join(quora.root, 'data', 'processed_all.pkl'))
#
# # drop nans from dataframe
# df.dropna(inplace=True)

cols = ['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim']

df = pd.read_pickle(os.path.join(quora.root, 'data', 'processed_v3_5000.pkl'))

# conditional coloring based one one variable (should be categorical)
sns.pairplot(df[['similarity', 'word_share', 'overlap', 'diff', 'word_diff', 'char_diff', 'cosine_sim', 'is_duplicate']],
             hue='is_duplicate')

sns.plt.show()

for col in cols:
    sns.plt.figure()
    # sns.distplot(df[df['is_duplicate'] == 1.0][col], color='red')
    # sns.distplot(df[df['is_duplicate'] == 0.0][col], color='green')

    sns.violinplot(x='is_duplicate', y=col, data=df)

sns.plt.show()
