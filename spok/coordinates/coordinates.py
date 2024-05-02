import numpy as np
import pandas as pd

from ..smath import norm as mnorm


def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.sin(theta) * np.cos(phi)
    return x, y, z

def cylindrical_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = r * np.sin(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(x / r)
    phi = np.arctan2(y, z)
    return r, theta, phi


def cartesian_to_parabolic(x, y, z, xf):
    r = np.sqrt((x - xf) ** 2 + y ** 2 + z ** 2)
    s = np.sqrt((x - xf) + r)
    t = np.sqrt(-(x - xf) + r)
    p = np.arctan2(z, y)
    return s, t, p


def choice_coordinate_system(R, theta, phi, **kwargs):
    coord_sys = kwargs.get('coord_sys','cartesian')
    if coord_sys == 'cartesian':
        return spherical_to_cartesian(R, theta, phi)
    elif coord_sys == 'spherical':
        return R, theta, phi
    else:
        print('Error : coord_sys parameter must be set to "cartesian" or "spherical" ')




def add_cst_radiuus(pos, cst,coord_sys):
    if coord_sys == 'cartesian':
        r, theta, phi = cartesian_to_spherical(pos.X, pos.Y, pos.Z)
    else:
        r, theta, phi = pos.R, pos.theta, pos.phi
    r = r + cst
    x, y, z = spherical_to_cartesian(r, theta, phi)
    return pd.DataFrame({'X': x, 'Y': y, 'Z': z})


def change_coordinate_system(x, y, z, x_uni, y_uni, z_uni):
    X = x_uni[:,0]*x + x_uni[:,1]*y + x_uni[:,2]*z
    Y = y_uni[:,0]*x + y_uni[:,1]*y + y_uni[:,2]*z
    Z = z_uni[:,0]*x + z_uni[:,1]*y + z_uni[:,2]*z
    return X, Y, Z

def gpim_base(vx, vy, vz, bx, by, bz):
    V = mnorm(vx, vy, vz)
    X = np.array([-vx / V, -vy / V, -vz / V])
    B_scal_X = bx * X[0] + by * X[1] + bz * X[2]
    Y = np.zeros_like(X)
    sign = np.sign(B_scal_X)

    if isinstance(bx, float) | isinstance(bx, int):
        if bx == 0:
            if by != 0:
                sign = -np.sign(by)
            else:
                sign = np.sign(bz)
    else:
        sign[(bx == 0) & (by != 0)] = -np.sign(by[(bx == 0) & (by != 0)])
        sign[(bx == 0) & (by == 0)] = np.sign(bz[(bx == 0) & (by == 0)])

    Y[1] = -sign * by + sign * B_scal_X * X[1]
    Y[2] = -sign * bz + sign * B_scal_X * X[2]

    Y_norm = mnorm(Y[0], Y[1], Y[2])
    Y = np.array([Y[0] / Y_norm, Y[1] / Y_norm, Y[2] / Y_norm])
    X, Y = X.T, Y.T
    Z = np.cross(X, Y)
    return X, Y, Z


def swi_base(vx, vy, vz, bx, by, bz):
    coa_zhang19 = np.arctan(bx / np.sqrt(by ** 2 + bz ** 2))
    sign = np.sign(coa_zhang19)
    if isinstance(coa_zhang19, float) or isinstance(coa_zhang19, int):
        if coa_zhang19 == 0:
            sign = np.sign(by)
            if by == 0:
                sign = np.sign(bz)
    else:
        sign[coa_zhang19 == 0] = np.sign(by[coa_zhang19 == 0])
        sign[(coa_zhang19 == 0) & (by == 0)] = np.sign(bz[(coa_zhang19 == 0) & (by == 0)])

    V = mnorm(vx, vy, vz)
    X = np.array([-vx / V, -vy / V, -vz / V])
    B = np.array([sign * bx, sign * by, sign * bz])
    Z = np.cross(X.T, B.T).T
    Z_norm = mnorm(Z[0], Z[1], Z[2])
    Z = np.array([Z[0] / Z_norm, Z[1] / Z_norm, Z[2] / Z_norm])
    X, Z = X.T, Z.T
    Y = np.cross(Z, X)
    return X, Y, Z

def rotates_from_phi_angle(x,y,z,angle):
    r,th,ph = cartesian_to_spherical(x,y,z)
    return spherical_to_cartesian(r,th,ph+angle)
