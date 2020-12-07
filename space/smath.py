import numpy as np
import pandas as pd

def norm(u, v, w):
    return np.sqrt(u**2 + v**2 + w**2)


def resolve_poly2(a, b, c, roots=None):
    delta = b ** 2 - 4 * a * c
    if isinstance(delta, np.ndarray) | isinstance(delta, pd.Series):
        a = np.ones(len(delta)) * a
        b = np.ones(len(delta)) * b
        c = np.ones(len(delta)) * c
        r1 = np.zeros(len(delta))
        r2 = np.zeros(len(delta))
        r1[(abs(a) <= 1e-7) & (abs(b) >= 1e-7)] = -c[(abs(a) <= 1e-7) & (abs(b) >= 1e-7)] / b[
            (abs(a) <= 1e-7) & (abs(b) >= 1e-7)]
        r2[(abs(a) <= 1e-7) & (abs(b) >= 1e-7)] = -c[(abs(a) <= 1e-7) & (abs(b) >= 1e-7)] / b[
            (abs(a) <= 1e-7) & (abs(b) >= 1e-7)]
        r1[abs(a) >= 1e-7] = (-b[abs(a) >= 1e-7] + np.sqrt(delta[abs(a) >= 1e-7])) / (2 * a[abs(a) >= 1e-7])
        r2[abs(a) >= 1e-7] = (-b[abs(a) >= 1e-7] - np.sqrt(delta[abs(a) >= 1e-7])) / (2 * a[abs(a) >= 1e-7])
        r1[delta < 0] = np.nan
        r2[delta < 0] = np.nan
        if roots == 0:
            return r1
        elif roots == 1:
            return r2
        else:
            return r1, r2

    else:
        if delta < 0:
            r1 = r2 = np.nan
        elif (abs(a) <= 1e-7) & (abs(b) >= 1e-7):
            r1 = r2 = -c / b
        elif (abs(a) >= 1e-7):
            r1 = (-b + np.sqrt(delta)) / (2 * a)
            r2 = (-b - np.sqrt(delta)) / (2 * a)
        else:
            r1 = r2 = 0

        if roots == 0:
            return r1
        elif roots == 1:
            return r2
        else:
            return r1, r2
