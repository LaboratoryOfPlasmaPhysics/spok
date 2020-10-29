import numpy as np


def listify(arg):
    if none_iterable(arg):
        return [arg]
    else:
        return arg
    
def none_iterable(*args):
    """
    return true if none of the arguments are either lists or tuples
    """
    return all([not isinstance(arg, list) and not isinstance(arg, tuple) and  not isinstance(arg, np.ndarray) for arg in args])   

def index_isin(df1,df2):
    return df1[df1.index.isin(df2.index)]
 
    
def same_index(a,b):
    a = index_isin(a,b)
    b = index_isin(b,a)
    return a,b

def eliminateNoneValidValues(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()

