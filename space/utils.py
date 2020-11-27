import numpy as np
import pandas as pd


def listify(arg):
    if none_iterable(arg):
        return [arg]
    else:
        return arg


def none_iterable(*args):
    """
    return true if none of the arguments are either lists or tuples
    """
    return all([not isinstance(arg, list) and not isinstance(arg, tuple) and not isinstance(arg,np.ndarray) and not isinstance(arg, pd.Series) for arg in args])

def index_isin(df1, df2):
    if isinstance(df1, list):
        return [df[df.index.isin(df2.index)] for df in df1]
    else:
        return df1[df1.index.isin(df2.index)]


def same_index(a,b):
    a = index_isin(a,b)
    b = index_isin(b,a)
    return a,b

def eliminate_none_valid_values(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def add_columns_to_df(dataframe,var,name):
    df = dataframe.copy()
    for n,v in zip(name,var):
        df[n]=v
    return df

def select_data_with_condition(data,cond):
    if isinstance(data, list):
        return [d[cond] for d in data]
    else :
        return data[cond]
