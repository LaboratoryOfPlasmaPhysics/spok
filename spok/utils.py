import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


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


def make_center_bins(vec, dd = 1):
    vec = listify(vec)
    if dd== 1 :
        return [0.5*(v[1:]+v[:-1]) for v in vec]
    elif dd==2 :
        return [0.5*(v[1:,1:]+v[:-1,:-1]) for v in vec]
    elif dd==3 :
        return [0.5*(v[1:,1:,1:]+v[:-1,:-1,:-1]) for v in vec]


def reshape_to_2Darrays(lst_arrays):
    lst_arrays = [np.array(listify(el)) for el in lst_arrays]
    if np.sum([lst_arrays[0].shape!=el.shape for el in lst_arrays[1:]]):
        raise ValueError('All elements of lst_arrays must have the same shape')
    if len(lst_arrays[0].shape)==1 :
        a2d = np.array(lst_arrays).T
        old_shape = np.array(lst_arrays).shape
    else :
        a2d = np.asarray(lst_arrays)
        old_shape = a2d.shape
        a2d = a2d.T.ravel().reshape(np.prod(old_shape[1:]),old_shape[0])
    return a2d, old_shape


def reshape_to_original_shape(a2d, old_shape):
    if a2d.shape[0]==1 :
        lst_arrays = a2d.T
    else :
        lst_arrays = np.asarray([a2d.reshape(old_shape[-1],np.prod(old_shape[:-1])).T[i::old_shape[0]] for i in range(old_shape[0])])
    return lst_arrays


def regular_grid_interpolation(x, y, qty, new_x, new_y, **kwargs):
    method = kwargs.get('method','linear')
    qty_2d = reshape_to_2Darrays([qty])[0]
    xy =  reshape_to_2Darrays([x,y])[0]
    reg_qty = griddata(xy, qty_2d[:,0], (new_x, new_y), method=method)
    return reg_qty

def nan_gaussian_filter(arr, sigma, mode='nearest'):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = gaussian_filter(
            loss, sigma=sigma, mode=mode, cval=1)

    gauss = arr / (1-loss)
    gauss[nan_msk] = 0
    gauss = gaussian_filter(
            gauss, sigma=sigma, mode=mode, cval=0)
    gauss[nan_msk] = np.nan

    return gauss


