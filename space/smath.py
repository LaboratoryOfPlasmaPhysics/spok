import numpy as np
import pandas as pd

def norm(u, v, w):
    return np.sqrt(u**2 + v**2 + w**2)

def resolve_poly2(a, b, c, roots=None):
    def fun(a, b, c):
        if roots is None:
            return np.roots([a, b, c])
        return np.roots([a,b,c])[roots]

    vfun = np.vectorize(fun, otypes=(np.ndarray,))
    return vfun(a, b, c)
