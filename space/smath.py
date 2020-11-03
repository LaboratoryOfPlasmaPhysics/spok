import numpy as np

def norm(u, v, w):
    return np.sqrt(u**2 + v**2 + w**2)


def resolve_poly2(a, b, c):
    if isinstance(a, np.ndarray):
        r1 = np.zeros_like(a)
        r2 = np.zeros_like(a)
        a_null = np.where(np.abs(a) < 1e-6)[0]

        delta = b ** 2 - 4 * a * c
        np.testing.assert_array_less(-delta, 0)

        r1, r2 = (-b + np.sqrt(delta)) / (2 * a), (-b + np.sqrt(delta)) / (2 * a)
        if isinstance(c, np.ndarray):
            c = c[a_null]
        if isinstance(b, np.ndarray):
            b = b[a_null]
        r1[a_null] = r2[a_null] = -c / b
    else:
        delta = b ** 2 - 4 * a * c
        np.testing.assert_array_less(-delta, 0)
        r1, r2 = (-b + np.sqrt(delta)) / (2 * a), (-b - np.sqrt(delta)) / (2 * a)
        if np.abs(a) < 1e-6:
            r1 = r2 = -c / b
    return r1, r2
