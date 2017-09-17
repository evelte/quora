
def get_cols_with_nans(df):
    for col in df.columns.values:
        if df[col].isnull().any():
            print(col, df[col].dtype, 'missing')
        else:
            print(col, df[col].dtype, 'no missing values')