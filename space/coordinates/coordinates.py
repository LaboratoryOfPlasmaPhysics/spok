import numpy as np
import pandas as pd


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
    phi[z==0]=np.sign(y)*np.pi/2
    return r, theta, phi


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


def swi_base(omni_data):
    norm_V = np.sqrt(omni_data.Vx ** 2 + (omni_data.Vy ) ** 2 + omni_data.Vz ** 2)
    X = np.array([-np.array([vx, vy, vz]) / V for vx, vy, vz, V in
                  zip(omni_data.Vx.values, omni_data.Vy.values, omni_data.Vz.values, norm_V)])
    B = np.array([np.array([bx, by, bz]) for bx, by, bz in zip(np.sign(omni_data.COA.values) * omni_data.Bx.values,
                                                               np.sign(omni_data.COA.values) * omni_data.By.values,
                                                               np.sign(omni_data.COA.values) * omni_data.Bz.values)])

    Z = np.cross(X, B)

    norm_cross = np.array(np.sqrt(Z[:, 0] ** 2 + Z[:, 1] ** 2 + Z[:, 2] ** 2))
    Z = np.array([z / n for z, n in zip(Z, norm_cross)])
    Y = np.cross(Z, X)
    return X, Y, Z


def to_swi(omni_data, msh_data, pos_msh):
    X_swi, Y_swi, Z_swi = swi_base(omni_data)
    data = msh_data.copy()
    o_data = omni_data.copy()
    pos = pos_msh.copy()



    o_data['Vx'] = X_swi[:,0]*omni_data['Vx']+X_swi[:,1]*(omni_data['Vy'])+X_swi[:,2]*omni_data['Vz']
    o_data['Vy'] = Y_swi[:,0]*omni_data['Vx']+Y_swi[:,1]*(omni_data['Vy'])+Y_swi[:,2]*omni_data['Vz']
    o_data['Vz'] = Z_swi[:,0]*omni_data['Vx']+Z_swi[:,1]*(omni_data['Vy'])+Z_swi[:,2]*omni_data['Vz']

    o_data['Bx'] = np.sign(omni_data['COA'])*(X_swi[:,0]*omni_data['Bx']+X_swi[:,1]*omni_data['By']+X_swi[:,2]*omni_data['Bz'])
    o_data['By'] = np.sign(omni_data['COA'])*(Y_swi[:,0]*omni_data['Bx']+Y_swi[:,1]*omni_data['By']+Y_swi[:,2]*omni_data['Bz'])
    o_data['Bz'] = np.sign(omni_data['COA'])*(Z_swi[:,0]*omni_data['Bx']+Z_swi[:,1]*omni_data['By']+Z_swi[:,2]*omni_data['Bz'])

    data['Vx'] = X_swi[:,0]*msh_data['Vx']+X_swi[:,1]*msh_data['Vy']+X_swi[:,2]*msh_data['Vz']
    data['Vy'] = Y_swi[:,0]*msh_data['Vx']+Y_swi[:,1]*msh_data['Vy']+Y_swi[:,2]*msh_data['Vz']
    data['Vz'] = Z_swi[:,0]*msh_data['Vx']+Z_swi[:,1]*msh_data['Vy']+Z_swi[:,2]*msh_data['Vz']

    data['Bx'] = np.sign(omni_data['COA'])*(X_swi[:,0]*msh_data['Bx']+X_swi[:,1]*msh_data['By']+X_swi[:,2]*msh_data['Bz'])
    data['By'] = np.sign(omni_data['COA'])*(Y_swi[:,0]*msh_data['Bx']+Y_swi[:,1]*msh_data['By']+Y_swi[:,2]*msh_data['Bz'])
    data['Bz'] = np.sign(omni_data['COA'])*(Z_swi[:,0]*msh_data['Bx']+Z_swi[:,1]*msh_data['By']+Z_swi[:,2]*msh_data['Bz'])


    pos['X'] = X_swi[:,0]*pos_msh['X']+X_swi[:,1]*pos_msh['Y']+X_swi[:,2]*pos_msh['Z']
    pos['Y'] = Y_swi[:,0]*pos_msh['X']+Y_swi[:,1]*pos_msh['Y']+Y_swi[:,2]*pos_msh['Z']
    pos['Z'] = Z_swi[:,0]*pos_msh['X']+Z_swi[:,1]*pos_msh['Y']+Z_swi[:,2]*pos_msh['Z']


    return data,pos,o_data

