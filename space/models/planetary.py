import numpy as np
import pandas as pd
from scipy import constants as cst
from datetime import timedelta, datetime

import sys

sys.path.append('.')
from ..coordinates import coordinates as coords
from ..smath import resolve_poly2_real_roots
from ..smath import norm as mnorm
from ..utils import listify




def get_tilt(date):
    doy = pd.to_numeric(pd.DatetimeIndex(date).strftime('%j'))
    ut = pd.to_numeric(pd.DatetimeIndex(date).strftime('%H')) + pd.to_numeric(pd.DatetimeIndex(date).strftime('%M'))/60
    tilt_year = 23.4 * np.cos((doy-172)*2*np.pi/365.25)
    tilt_day = 11.2 * np.cos((ut-16.72)*2*np.pi/24)
    tilt = (tilt_year + tilt_day)*np.pi/180
    return tilt


def associate_SW_Safrankova(X_sat, omni, BS_standoff, dtm=0,sampling_time='5S',vx_median =-406.2):
    if dtm != 0:
        #vxmean = abs(omni.Vx.rolling(dt,min_periods=1).mean())
        vxmean = abs(omni.Vx.rolling(int((2*dtm+1)*timedelta(minutes=1)/(omni.index[-1]-omni.index[-2])),center=True,min_periods=1).mean())
    else:
        vxmean = abs(omni.Vx)
    BS_x0 = BS_standoff[BS_standoff.index.isin(X_sat.index)]
    BS_x0 = BS_x0.fillna(13.45)
    lag = np.array(np.round((BS_x0.values-X_sat.values)*6371/vx_median),dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time)
    vx = pd.Series(name='Vx',dtype=float)
    vx  = vx.append(vxmean.loc[time],ignore_index=True).fillna(abs(vx_median)).values
    lag = np.array(np.round((BS_x0.values-X_sat.values)*6371/vx),dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time)
    OMNI = pd.DataFrame(columns=omni.columns)
    OMNI = OMNI.append(omni.loc[time], ignore_index=True)
    OMNI.index = X_sat.index
    return OMNI.dropna()


def parabolic_magnetic_field_to_cartesian(Bs, Bt, Bp, s, t, p, hs, ht, hp):
    Bx = Bs * (s / hs) - Bt * (t / ht)
    By = Bs * (t / hs) * np.cos(p) + Bt * (s / ht) * np.cos(p) - Bp * np.sin(p)
    Bz = Bs * (t / hs) * np.sin(p) + Bt * (s / ht) * np.sin(p) + Bp * np.cos(p)
    return Bx, By, Bz


def KF1994(x, y, z, x0, x1, B0x, B0y, B0z):
    '''
    	x0 : standoff distance MP
    	x1 : standoff distance BS
    '''
    xf = x0/2
    s, t, p = coords.cartesian_to_parabolic(x, y, z, xf)
    s0 = np.sqrt(2 * (x0 - xf))
    s1 = np.sqrt(2 * (x1 - xf))
    c = s1 ** 2 / (s1 ** 2 - s0 ** 2)
    k1 = B0x * c
    k2 = (B0y * np.cos(p) + B0z * np.sin(p)) * c
    hs = ht = np.sqrt(s ** 2 + t ** 2)
    hp = s * t

    Bs = (1 / hs) * (k1 * (s - s0 ** 2 / s) + k2 * t * (1 - s0 ** 2 / s ** 2))
    Bt = (1 / ht) * (-k1 * t + k2 * (s + s0 ** 2 / s))
    Bp = (1 / hp) * ((-B0y * np.sin(p) + B0z * np.cos(p)) * c * (s + s0 ** 2 / s) * t)

    Bx, By, Bz = parabolic_magnetic_field_to_cartesian(Bs, Bt, Bp, s, t, p, hs, ht, hp)
    return Bx, By, Bz




def _formisano1979(theta, phi, **kwargs):
    a11, a22, a33, a12, a13, a23, a14, a24, a34, a44 = kwargs["coefs"]

    if 'Pd' in kwargs:
        a14 = a14 * (2.056 / kwargs['Pd']) ** (1 / 6)
        a44 = a44 * (2.056 / kwargs['Pd']) ** (1 / 3)
    x = np.cos(theta)
    y = np.sin(theta)*np.sin(phi)
    z = np.sin(theta)*np.cos(phi)
    A = a11 * x ** 2 + a22 * y ** 2 + a33 * z ** 2 + a12 * x * y + a13 * x * z + a23 * y * z
    B = a14 * x + a24 * y + a34 * z
    C = a44
    D = B ** 2 - 4 * A * C
    return (-B + np.sqrt(D)) / (2 * A)




def formisano1979(theta, phi, **kwargs):
    '''
    Formisano 1979 magnetopause model. Give the default position of the magnetopause.
    function's arguments :
        - theta : angle in radiant, can be int, float or array (1D or 2D)
        - phi   : angle in radiant, can be int, float or array (1D or 2D)
    kwargs:
        - boundary  : "magnetopause", "bow_shock"
        - coord_sys  : can be "cartesian" (default) or "spherical"

    information : to get a particular point theta and phi must be an int or a float
    (ex : the nose of the boundary is given with the input theta=0 and phi=0). If a plan (2D) of
    the boundary is wanted one of the two angle must be an array and the other one must be
    an int or a float (ex : (XY) plan with Z=0 is given with the input
    theta=np.linespace(-np.pi/2,np.pi/2,100) and phi=0). To get the 3D boundary theta and phi
    must be given an array, if it is two 1D array a meshgrid will be performed to obtain a two 2D array.

    return : X,Y,Z (coord_sys="cartesian")or R,theta,phi (coord_sys="spherical") depending on the chosen coord_sys
    '''

    if kwargs["boundary"] == "magnetopause":
        coefs = [0.65, 1, 1.16, 0.03, -0.28, -0.11, 21.41, 0.46, -0.36, -221] # coeff magnetopause  Romashets 2019 with aberration, take into account earth's rotation
        #coefs = [0.66, 1, 1.16, 0.08, -0.29, -0.09, 21.47, -0.97, -0.36, -222] # coeff magnetopause Romashets 2019 without aberation
    elif kwargs["boundary"] == "bow_shock":
        coefs = [0.52, 1, 1.05, 0.13, -0.16, -0.08, 47.53, -0.42, 0.67, -613] # coeff bow shock  Romashets 2019 with aberration, take into account earth's rotation
        #coefs = [0.54,1,1.06,0.19,-0.17,-0.07,47.90,-3.62,0.68,-619] # coeff bow shock  Romashets 2019 without aberation
    else:
        raise ValueError("boundary: {} not allowed".format(kwargs["boundary"]))


    r = _formisano1979(theta, phi, coefs=coefs,**kwargs)

    return coords.choice_coordinate_system(r, theta, phi, **kwargs)

def mp_formisano1979(theta, phi, **kwargs):
    return formisano1979(theta, phi, boundary="magnetopause", **kwargs)


def bs_formisano1979(theta, phi, **kwargs):
    return formisano1979(theta, phi, boundary="bow_shock", **kwargs)


def Fairfield1971(x, args):
    '''
    Fairfield 1971 : Magnetopause and Bow shock models. Give positions of the boudaries in plans (XY) with Z=0 and (XZ) with Y=0.
    function's arguments :
        - x :  X axis (array) in Re (earth radii)
        - args : coefficients Aij are determined from many boundary crossings and they depend on upstream conditions.

        --> Default parameter for the bow shock and the magnetopause respectively are :
            default_bs_fairfield = [0.0296,-0.0381,-1.28,45.644,-652.1]
            default_mp_fairfield = [-0.0942,0.3818,0.498,17.992,-248.12]

     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the wanted boudary to plot (XY) and (XZ) plans.
    '''

    A, B, C, D, E = args[0], args[1], args[2], args[3], args[4]

    a = 1
    b = A * x + C
    c = B * x ** 2 + D * x + E

    delta = b ** 2 - 4 * a * c

    ym = (-b - np.sqrt(delta)) / (2 * a)
    yp = (-b + np.sqrt(delta)) / (2 * a)

    pos = pd.DataFrame({'X': np.concatenate([x, x[::-1]]),
                        'Y': np.concatenate([yp, ym[::-1]]),
                        'Z': np.concatenate([yp, ym[::-1]]), })

    return pos.dropna()


def bs_Jerab2005(theta, phi, **kwargs):
    '''
    Jerab 2005 Bow shock model. Give positions of the box shock in plans (XY) with Z=0 and (XZ) with Y=0 as a function of the upstream solar wind.
    function's arguments :
        - Np : Proton density of the upstream conditions
        - V  : Speed of the solar wind
        - B  : Intensity of interplanetary magnetic field
        - gamma : Polytropic index ( default gamma=2.15)


        --> mean parameters :  Np=7.35, V=425.5,  B=5.49

     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the bow shock to plot (XY) and (XZ) plans.
    '''

    def make_Rav(theta, phi):
        a11 = 0.45
        a22 = 1
        a33 = 0.8
        a12 = 0.18
        a14 = 46.6
        a24 = -2.2
        a34 = -0.6
        a44 = -618

        x = np.cos(theta)
        y = np.sin(theta) * np.sin(phi)
        z = np.sin(theta) * np.cos(phi)

        a = a11 * x ** 2 +  a22 * y ** 2 + a33 * z ** 2 + a12 * x * y
        b = a14 * x + a24 * y + a34 * z
        c = a44

        delta = b ** 2 - 4 * a * c

        R = (-b + np.sqrt(delta)) / (2 * a)
        return R

    Np = kwargs.get('Np', 6.025)
    V = kwargs.get('V', 427.496)
    B = kwargs.get('B', 5.554)
    gamma = kwargs.get('gamma', 5./3)
    Ma = V * 1e3 * np.sqrt(Np * 1e6 * cst.m_p * cst.mu_0) / (B * 1e-9)

    C = 91.55
    D = 0.937 * (0.846 + 0.042 * B)
    R0 = make_Rav(0, 0)

    Rav = make_Rav(theta, phi)
    K = ((gamma - 1) * Ma ** 2 + 2) / ((gamma + 1) * (Ma ** 2 - 1))
    r = (Rav / R0) * (C / (Np * V ** 2) ** (1 / 6)) * (1 + D * K)

    return coords.choice_coordinate_system(r, theta, phi, **kwargs)


def mp_shue1997(theta, phi, **kwargs):
    Pd = kwargs.get("Pd", 2.056)
    Bz = kwargs.get("Bz", -0.001)

    if isinstance(Bz, float) | isinstance(Bz, int):
        if Bz >= 0:
            r0 = (11.4 + 0.13 * Bz) * Pd ** (-1 / 6.6)
        else:
            r0 = (11.4 + 0.14 * Bz) * Pd ** (-1 / 6.6)
    else:
        if isinstance(Pd, float) | isinstance(Pd, int):
            Pd = np.ones_like(Bz) * Pd
        r0 = (11.4 + 0.13 * Bz) * Pd ** (-1 / 6.6)
        r0[Bz < 0] = (11.4 + 0.14 * Bz[Bz < 0]) * Pd[Bz < 0] ** (-1 / 6.6)
    a = (0.58 - 0.010 * Bz) * (1 + 0.010 * Pd)
    r = r0 * (2. / (1 + np.cos(theta))) ** a
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)


def mp_shue1998(theta, phi, **kwargs):
    '''
    Shue 1998 Magnetopause model.

     Returns the MP distance for given theta,
     dynamic pressure (in nPa) and Bz (in nT).

    - theta : Angle from the x axis (model is cylindrical symmetry)
         example : theta = np.arange( -np.pi+0.01, np.pi-0.01, 0.001)
    - phi   : unused, azimuthal angle, to get the same interface as all models

    kwargs:
    - "pdyn": Dynamic Pressure in nPa
    - "Bz"  : z component of IMF in nT

    '''

    Pd = kwargs.get("Pd", 2.056)
    Bz = kwargs.get("Bz", -0.001)

    r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd ** (-1. / 6.6)
    a = (0.58 - 0.007 * Bz) * (1 + 0.024 * np.log(Pd))
    r = r0 * (2. / (1 + np.cos(theta))) ** a
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)


def mp_lin2010(theta, phi, **kwargs):
    ''' The Lin 2010 Magnetopause model. Returns the MP distance for a given
    azimuth (phi), zenith (th), solar wind dynamic and magnetic pressures (nPa)
    and Bz (in nT).
    * th: Zenith angle from positive x axis (zenith) (between 0 and pi)
    * phi: Azimuth angle from y axis, about x axis (between 0 abd 2 pi)
    * Pd: Solar wind dynamic pressure in nPa
    * Pm: Solar wind magnetic pressure in nPa
    * tilt: Dipole tilt
    '''
    a = [12.544, -0.194, 0.305, 0.0573, 2.178, 0.0571, -0.999, 16.473, 0.00152, 0.382, 0.0431, -0.00763, -0.210, 0.0405,
         -4.430, -0.636, -2.600, 0.832, -5.328, 1.103, -0.907, 1.450]

    if isinstance(theta, np.ndarray) | isinstance(theta, pd.Series):
        t = np.array(theta).copy()
        idx = np.where(t < 0)[0]
        if isinstance(phi, np.ndarray) | isinstance(phi, pd.Series):
            p = np.array(phi).copy()
            p[idx] = p[idx] + np.pi
        else:
            p = phi * np.ones(t.shape)
            p[idx] = p[idx] + np.pi
    else:
        t = theta
        p = phi + (t < 0) * np.pi

    t = np.sign(t) * t

    Pd = kwargs.get('Pd', 2.056)
    Bx = kwargs.get('Bx', 0.032)
    By = kwargs.get('By', -0.015)
    Bz = kwargs.get('Bz', -0.001)
    tilt = kwargs.get('tilt', 0.)

    B_param = [('Bx' in kwargs), ('By' in kwargs), ('Bz' in kwargs)]
    if all(B_param):
        Pm = (Bx ** 2 + By ** 2 + Bz ** 2) * 1e-18 / (2 * cst.mu_0) * 1e9
    elif not any(B_param):
        Pm = 0.016
    else:
        raise ValueError('None or all Bx, By, Bz parameters must be set')

    P = Pd + Pm

    def exp2(i):
        return a[i] * (np.exp(a[i + 1] * Bz) - 1) / (np.exp(a[i + 2] * Bz) + 1)

    def quad(i, s):
        return a[i] + s[0] * a[i + 1] * tilt + s[1] * a[i + 2] * tilt ** 2

    r0 = a[0] * P ** a[1] * (1 + exp2(2))

    beta = [a[6] + exp2(7),
            a[10],
            quad(11, [1, 0]),
            a[13]]

    f = np.cos(0.5 * t) + a[5] * np.sin(2 * t) * (1 - np.exp(-t))
    s = beta[0] + beta[1] * np.sin(p) + beta[2] * np.cos(p) + beta[3] * np.cos(p) ** 2
    f = f ** (s)

    c = {}
    d = {}
    TH = {}
    PHI = {}
    e = {}
    for i, s in zip(['n', 's'], [1, -1]):
        c[i] = a[14] * P ** a[15]
        d[i] = quad(16, [s, 1])
        TH[i] = quad(19, [s, 0])
        PHI[i] = np.cos(t) * np.cos(TH[i])
        PHI[i] = PHI[i] + np.sin(t) * np.sin(TH[i]) * np.cos(p - (1 - s) * 0.5 * np.pi)
        PHI[i] = np.arccos(PHI[i])
        e[i] = a[21]
    r = f * r0

    Q = c['n'] * np.exp(d['n'] * PHI['n'] ** e['n'])
    Q = Q + c['s'] * np.exp(d['s'] * PHI['s'] ** e['s'])

    r = r + Q

    return coords.choice_coordinate_system(r, theta, phi, **kwargs)


def mp_liu2015(theta, phi, **kwargs):
    if isinstance(theta, np.ndarray) | isinstance(theta, pd.Series):
        t = np.array(theta).copy()
        idx = np.where(t < 0)[0]
        if isinstance(phi, np.ndarray) | isinstance(phi, pd.Series):
            p = np.array(phi).copy()
            p[idx] = p[idx] + np.pi
        else:
            p = phi * np.ones(t.shape)
            p[idx] = p[idx] + np.pi
    else:
        t = theta
        p = phi + (t < 0) * np.pi

    t = np.sign(t) * t

    Pd = kwargs.get('Pd', 2.056)
    Bx = kwargs.get('Bx', 0.032)
    By = kwargs.get('By', -0.015)
    Bz = kwargs.get('Bz', -0.001)
    tilt = kwargs.get('tilt', 0.)
    B_param = [('Bx' in kwargs), ('By' in kwargs), ('Bz' in kwargs)]
    if all(B_param):
        Pm = (Bx ** 2 + By ** 2 + Bz ** 2) * 1e-18 / (2 * cst.mu_0) * 1e9
    elif not any(B_param):
        Pm = 0.016
    else:
        raise ValueError('None or all Bx, By, Bz parameters must be set')

    P = Pd + Pm

    r0 = (10.56 + 0.956 * np.tanh(0.1795 * (Bz + 10.78))) * P ** (-0.1699)

    alpha_0 = (0.4935 + 0.1095 * np.tanh(0.07217 * (Bz + 6.882))) * (1 + 0.01182 * np.log(Pd))

    alpha_z = 0.06263 * np.tanh(0.0251 * tilt)

    alpha_p = (0.06354 + 0.07764 * np.tanh(0.07217 * (abs(Bz) + 4.851))) * (1 + 0.01182 * np.log(Pd))

    delta_alpha = 0.02582 * np.tanh(0.0667 * Bx) * np.sign(Bx)

    if isinstance(Bz, np.ndarray) | isinstance(Bz, pd.Series):
        idx_zero = np.where(Bz != 0)[0]
        omega = np.zeros_like(np.shape(Bz))
        omega = omega + np.sign(By) * np.pi / 2
        omega[idx_zero] = np.arctan2(0.1718 * By[idx_zero] * (By[idx_zero] ** 2 + Bz[idx_zero] ** 2) ** 0.194,
                                     Bz[idx_zero])

    else:
        if Bz != 0:
            omega = np.arctan2(0.1718 * By * (By ** 2 + Bz ** 2) ** 0.194, Bz)
        else:
            omega = np.sign(By) * np.pi / 2

    alpha = alpha_0 + alpha_z * np.cos(p) + (alpha_p + delta_alpha * np.sign(np.cos(p))) * np.cos(
        2 * (p - omega))

    l_n = (0.822 + 0.2921 * np.tanh(0.08792 * (Bz + 10.12))) * (1 - 0.01278 * tilt)
    l_s = (0.822 + 0.2921 * np.tanh(0.08792 * (Bz + 10.12))) * (1 + 0.01278 * tilt)
    w = (0.2382 + 0.005806 * np.log(Pd)) * (1 + 0.0002335 * tilt ** 2)

    C = np.exp(-abs(t - l_n) / w) * (1 + np.sign(np.cos(p))) + np.exp(-abs(t - l_s) / w) * (
        1 + np.sign(-np.cos(p)))

    r = (r0 * (2 / (1 + np.cos(t))) ** alpha) * (1 - 0.1 * C * np.cos(p) ** 2)

    return coords.choice_coordinate_system(r, theta, phi, **kwargs)



def mp_jelinek2012(theta, phi, **kwargs):
    lamb = 1.54
    R = 12.82
    epsilon = 5.26
    Pd = kwargs.get('Pd', 2.056)

    R0 = 2 * R * Pd ** (-1 / epsilon)
    r = R0 / (np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + np.sin(theta) * np.sin(theta) * lamb ** 2))
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)


def bs_jelinek2012(theta, phi, **kwargs):
    lamb = 1.17
    R = 15.02
    epsilon = 6.55
    Pd = kwargs.get('Pd', 2.056)

    R0 = 2 * R * Pd ** (-1 / epsilon)
    r = R0 / (np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + np.sin(theta) * np.sin(theta) * lamb ** 2))
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)

def mp_nguyen2020(theta, phi, **kwargs):
    def inv_cos(t):
        return 2 / (1 + np.cos(t))

    ## return an array of phi and theta in the right ranges of values
    if isinstance(theta, np.ndarray) | isinstance(theta, pd.Series):
        t = np.array(theta).copy()
        idx = np.where(t < 0)[0]
        if isinstance(phi, np.ndarray) | isinstance(phi, pd.Series):
            p = np.array(phi).copy()
            p[idx] = p[idx] + np.pi
        else:
            p = phi * np.ones(t.shape)
            p[idx] = p[idx] + np.pi
    else:
        t = theta
        p = phi + (t < 0) * np.pi

    t = np.sign(t) * t

    # coefficients
    a = [10.61,
         -0.150,
         0.027,
         0.207,
         1.62,
         0.558,
         0.135,
         0.0150,
         -0.0839]

    Pd = kwargs.get('Pd', 2.056)
    Bx = kwargs.get('Bx', 0.032)
    By = kwargs.get('By', -0.015)
    Bz = kwargs.get('Bz', -0.001)
    gamma = kwargs.get('tilt', 0.)

    B_param = [('Bx' in kwargs), ('By' in kwargs), ('Bz' in kwargs)]
    if all(B_param):
        Pm = (Bx ** 2 + By ** 2 + Bz ** 2) * 1e-18 / (2 * cst.mu_0) * 1e9
    elif not any(B_param):
        Pm = 0.016
    else:
        raise ValueError('None or all Bx, By, Bz parameters must be set')

    omega = np.sign(By) * np.arccos(Bz / np.sqrt(By ** 2 + Bz ** 2))
    P = Pd + Pm
    r0 = a[0] * (1 + a[2] * np.tanh(a[3] * Bz + a[4])) * P ** (a[1])

    alpha0 = a[5]
    alpha1 = a[6] * gamma
    alpha2 = a[7] * np.cos(omega)
    alpha3 = a[8] * np.cos(omega)

    alpha = alpha0 + alpha1 * np.cos(p) + alpha2 * np.sin(p) ** 2 + alpha3 * np.cos(p) ** 2
    r = r0 * inv_cos(t) ** alpha
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)


def mp_sibeck1991(theta, phi, **kwargs):
    Pd = kwargs.get('Pd', 2.056)
    if (Pd >= 0.54) and (Pd < 0.87):
        a0 = 0.19
        b0 = 19.3
        c0 = -272.4
        p0 = 0.71
    elif (Pd >= 0.87) and (Pd < 1.47):
        a0 = 0.19
        b0 = 18.7
        c0 = -243.9
        p0 = 1.17
    elif (Pd >= 1.47) and (Pd < 2.60):
        a0 = 0.14
        b0 = 18.2
        c0 = -217.2
        p0 = 2.04
    elif (Pd >= 2.60) and (Pd < 4.90):
        a0 = 0.15
        b0 = 17.3
        c0 = -187.4
        p0 = 3.75
    elif (Pd >= 4.90) and (Pd < 9.90):
        a0 = 0.18
        b0 = 14.2
        c0 = -139.2
        p0 = 7.4

    else:
        raise ('Dynamic pressure none valid')

    a = a0 * np.cos(theta) ** 2 + np.sin(theta) ** 2
    b = b0 * np.cos(theta) * (p0 / Pd) ** (1 / 6)
    c = c0 * (p0 / Pd) ** (1 / 3)

    r = resolve_poly2_real_roots(a, b, c)[0]

    return coords.choice_coordinate_system(r, theta, phi, **kwargs)



def derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp):
    dxdt = drdt * np.cos(theta) - r * np.sin(theta)
    dydt = drdt * np.sin(theta) * np.sin(phi) + r * np.cos(theta) * np.sin(phi)
    dzdt = drdt * np.sin(theta) * np.cos(phi) + r * np.cos(theta) * np.cos(phi)

    normt = mnorm(dxdt, dydt, dzdt)
    normt[normt == 0] = 1

    dxdp = 0
    dydp = drdp * np.sin(theta) * np.sin(phi) + r * np.sin(theta) * np.cos(phi)
    dzdp = drdp * np.sin(theta) * np.cos(phi) - r * np.sin(theta) * np.sin(phi)

    normp = mnorm(dxdp, dydp, dzdp)
    normp[normp == 0] = 1

    return [dxdt / normt, dydt / normt, dzdt / normt], [dxdp / normp, dydp / normp, dzdp / normp]


def find_normal_from_tangents(tth, tph):
    [dxdt, dydt, dzdt], [dxdp, dydp, dzdp] = tth, tph
    pvx = dzdt * dydp - dydt * dzdp
    pvy = dxdt * dzdp - dzdt * dxdp
    pvz = dydt * dxdp - dxdt * dydp

    norm = mnorm(pvx, pvy, pvz)

    pvx[norm == 0] = 1
    norm[norm == 0] = 1

    return (pvx / norm, pvy / norm, pvz / norm)




def mp_sibeck1991_tangents(theta, phi, **kwargs):
    theta = listify(theta)
    phi = listify(phi)
    Pd = kwargs.get("Pd", 2.056)
    if (Pd >= 0.54) and (Pd < 0.87):
        a0 = 0.19
        b0 = 19.3
        c0 = -272.4
        p0 = 0.71
    elif (Pd >= 0.87) and (Pd < 1.47):
        a0 = 0.19
        b0 = 18.7
        c0 = -243.9
        p0 = 1.17
    elif (Pd >= 1.47) and (Pd < 2.60):
        a0 = 0.14
        b0 = 18.2
        c0 = -217.2
        p0 = 2.04
    elif (Pd >= 2.60) and (Pd < 4.90):
        a0 = 0.15
        b0 = 17.3
        c0 = -187.4
        p0 = 3.75
    elif (Pd >= 4.90) and (Pd < 9.90):
        a0 = 0.18
        b0 = 14.2
        c0 = -139.2
        p0 = 7.4

    else:
        raise ('Dynamic pressure none valid')

    a = a0 * np.cos(theta) ** 2 + np.sin(theta) ** 2
    dadt = 2 * np.cos(theta) * np.sin(theta) * (1 - a0)

    b = b0 * np.cos(theta) * (p0 / Pd) ** (1 / 6)
    dbdt = -b0 * np.sin(theta) * (p0 / Pd) ** (1 / 6)

    c = c0 * (p0 / Pd) ** (1 / 3)
    dcdt = 0

    delta = b ** 2 - 4 * a * c
    ddeltadt = 2 * b * dbdt - 4 * dadt * c

    u = -b + np.sqrt(delta)
    dudt = -dbdt + ddeltadt / (2 * np.sqrt(delta))

    v = 2 * a
    dvdt = 2 * dadt

    r = resolve_poly2_real_roots(a, b, c)[0]
    drdt = (dudt * v - dvdt * u) / v ** 2
    drdp = 0

    return derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp)


def mp_sibeck1991_normal(theta, phi, **kwargs):
    tth, tph = mp_sibeck1991_tangents(theta, phi, **kwargs)
    return find_normal_from_tangents(tth, tph)


def mp_shue1998_tangents(theta, phi, **kwargs):
    theta = listify(theta)
    phi = listify(phi)
    Pd = kwargs.get("Pd", 2.056)
    Bz = kwargs.get("Bz", -0.001)

    r0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd ** (-1. / 6.6)
    a = (0.58 - 0.007 * Bz) * (1 + 0.024 * np.log(Pd))
    r = r0 * (2. / (1 + np.cos(theta))) ** a
    drdt = r0 * a * (2 ** a) * np.sin(theta) / (1 + np.cos(theta)) ** (a + 1)
    drdp = 0
    return derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp)


def bs_jelinek2012_tangents(theta, phi, **kwargs):
    theta = listify(theta)
    phi = listify(phi)
    lamb = 1.17
    R = 15.02
    epsilon = 6.55
    Pd = kwargs.get('Pd', 2.056)

    R0 = 2 * R * Pd ** (-1 / epsilon)
    r = R0 / (np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2))
    test = R0 * np.sin(theta) * (
        np.cos(theta) * (1 - lamb ** 2) / np.sqrt(np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2)) / (
               np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2)) ** 2

    u0 = R0
    v0 = np.cos(theta) + np.sqrt(np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2)
    v0p = -np.sin(theta) + (np.cos(theta) * (-np.sin(theta)) + lamb ** 2 * np.sin(theta) * np.cos(theta)) / np.sqrt(
        np.cos(theta) ** 2 + (lamb * np.sin(theta)) ** 2)

    drdt = -u0 * v0p / v0 ** 2
    drdp = 0

    return derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp)


def mp_shue1998_normal(theta, phi, **kwargs):
    tth, tph = mp_shue1998_tangents(theta, phi, **kwargs)
    return find_normal_from_tangents(tth, tph)


def bs_jelinek2012_normal(theta, phi, **kwargs):
    tth, tph = bs_jelinek2012_tangents(theta, phi, **kwargs)
    return find_normal_from_tangents(tth, tph)


_models = { "mp_shue1998": mp_shue1998,
            "mp_shue1997": mp_shue1997,
            "mp_formisano1979": mp_formisano1979,
            "mp_liu2015" : mp_liu2015,
            "mp_jelinek2012" : mp_jelinek2012,
            "mp_lin2010" : mp_lin2010,
            "mp_nguyen2020" : mp_nguyen2020,
            "mp_sibeck1991" : mp_sibeck1991,
            "bs_formisano1979": bs_formisano1979,
            "bs_jerab2005": bs_Jerab2005,
            "bs_jelinek2012": bs_jelinek2012}

_tangents = {"mp_shue1998" : mp_shue1998_tangents,
            "mp_sibeck1991" : mp_sibeck1991_tangents,
             "bs_jelinek2012" : bs_jelinek2012_tangents,
            "mp_shue1997": None,
             "mp_formisano1979": None,
             "mp_liu2015": None,
             "mp_jelinek2012": None,
             "mp_lin2010": None,
             "mp_nguyen2020": None,
             "bs_formisano1979": None,
             "bs_jerab2005": None}

_normal = {"mp_shue1998" : mp_shue1998_normal,
            "mp_sibeck1991" : mp_sibeck1991_normal,
            "bs_jelinek2012" : bs_jelinek2012_normal,
            "mp_shue1997": None,
           "mp_formisano1979": None,
           "mp_liu2015": None,
           "mp_jelinek2012": None,
           "mp_lin2010": None,
           "mp_nguyen2020": None,
           "bs_formisano1979": None,
           "bs_jerab2005": None
           }

def available(model):
    if model == "magnetopause":
        return tuple([m for m in _models.keys() if m.startswith("mp_")])
    elif model == "bow_shock":
        return tuple([m for m in _models.keys() if m.startswith("bs_")])
    raise ValueError("invalid model type")




def _interest_points(model, **kwargs):
    dup = kwargs.copy()
    dup["coord_sys"] = "cartesian"
    x = model(0, 0, **dup)[0]
    y = ( abs(model(np.pi / 2, np.pi/2, **dup)[1]) + abs(model(np.pi / 2, -np.pi/2, **dup)[1]))/2
    xf = x - y ** 2 / (4 * x)
    return x, y, xf

def _parabolic_approx(theta, phi, x0, xf, **kwargs): # x0= nose boundary , xf= focal point
    K = x0 - xf
    a = np.sin(theta) ** 2
    b = 4 * K * np.cos(theta)
    c = -4 * K * x0
    r = resolve_poly2_real_roots(a, b, c)[0]
    return coords.choice_coordinate_system(r, theta, phi, **kwargs)



def check_parabconfoc(func):
    def wrapper(self, theta, phi, **kwargs):
        kwargs["parabolic"] = kwargs.get("parabolic", False)
        kwargs["confocal"] = kwargs.get("confocal", False)
        if kwargs["parabolic"] is False and kwargs["confocal"] is True:
            raise ValueError("cannot be confocal if not parabolic")
        return func(self, theta, phi, **kwargs)

    return wrapper


class Magnetosheath:
    def __init__(self, **kwargs):

        kwargs["magnetopause"] = kwargs.get("magnetopause", "mp_shue1998")
        kwargs["bow_shock"] = kwargs.get("bow_shock", "bs_jelinek2012")

        if not kwargs["magnetopause"].startswith("mp_") \
            or not kwargs["bow_shock"].startswith("bs_"):
            raise ValueError("invalid model name")

        self._magnetopause = _models[kwargs["magnetopause"]]
        self._bow_shock = _models[kwargs["bow_shock"]]
        self.model_magnetopause = kwargs["magnetopause"]
        self.model_bow_shock = kwargs["bow_shock"]
        self._tangents_magnetopause = _tangents[kwargs["magnetopause"]]
        self._normal_magnetopause = _normal[kwargs["magnetopause"]]
        self._tangents_bow_shock = _tangents[kwargs["bow_shock"]]
        self._normal_bow_shock = _normal[kwargs["bow_shock"]]



    @check_parabconfoc
    def magnetopause(self, theta, phi, **kwargs):
        if kwargs["parabolic"]:
            return self._parabolize(theta, phi, **kwargs)[0]
        else:
            if kwargs.get('normal', False) or kwargs.get('tangents', False) :
                ret_vectors = []
                ret_vectors.append(self._magnetopause(theta, phi, **kwargs))
                if kwargs.get('normal', False):
                    ret_vectors.append(self._normal_magnetopause(listify(theta), listify(phi), **kwargs))
                if kwargs.get('tangents', False):
                    ret_vectors.append(self._tangents_magnetopause(listify(theta), listify(phi), **kwargs))
                return ret_vectors
            else :
                return self._magnetopause(theta, phi, **kwargs)

    @check_parabconfoc
    def bow_shock(self, theta, phi, **kwargs):
        if kwargs["parabolic"]:
            return self._parabolize(theta, phi, **kwargs)[1]
        else:
            if kwargs.get('normal', False) or kwargs.get('tangents', False):
                ret_vectors = []
                ret_vectors.append(self._bow_shock(theta, phi, **kwargs))
                if kwargs.get('normal',False):
                    ret_vectors.append(self._normal_bow_shock(listify(theta), listify(phi), **kwargs))
                if kwargs.get('tangents',False):
                    ret_vectors.append(self._tangents_bow_shock(listify(theta), listify(phi), **kwargs))
                return ret_vectors
            else :
                return self._bow_shock(theta, phi, **kwargs)


    @check_parabconfoc
    def boundaries(self, theta, phi, **kwargs):
        if kwargs["parabolic"]:
            return self._parabolize(theta, phi, **kwargs)
        else:
            return self._magnetopause(theta, phi, **kwargs), \
                   self._bow_shock(theta, phi, **kwargs)

    def tangents_magnetopause(self, theta, phi, **kwargs):
        return self._tangents_magnetopause(listify(theta), listify(phi), **kwargs)

    def normal_magnetopause(self, theta, phi, **kwargs):
        return self._normal_magnetopause(listify(theta), listify(phi), **kwargs)


    def tangents_bow_shock(self, theta, phi,**kwargs):
        return self._tangents_bow_shock(listify(theta), listify(phi), **kwargs)

    def normal_bow_shock(self, theta, phi,**kwargs):
        return self._normal_bow_shock(listify(theta), listify(phi), **kwargs)


    def _parabolize(self, theta, phi, **kwargs):
        xmp, y, xfmp = _interest_points(self._magnetopause, **kwargs)
        xbs, y, xfbs = _interest_points(self._bow_shock, **kwargs)
        if kwargs.get("confocal", False) is True:
            xfmp = xmp / 2
            xfbs = xmp / 2
        mp_coords = _parabolic_approx(theta, phi, xmp, xfmp, **kwargs)
        bs_coords = _parabolic_approx(theta, phi, xbs, xfbs, **kwargs)
        return mp_coords, bs_coords
