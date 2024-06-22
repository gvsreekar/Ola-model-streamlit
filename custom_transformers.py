import pandas as pd

def column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(lambda col: col.split('__')[-1], axis='columns')

import scipy.stats

def yeojohnson(df: pd.DataFrame) -> pd.DataFrame:
    df['Total Business Value'], _ = scipy.stats.yeojohnson(df['Total Business Value'])
    return df

def boxcox(df: pd.DataFrame) -> pd.DataFrame:
    df['Income'], _ = scipy.stats.boxcox(df['Income'])
    return df