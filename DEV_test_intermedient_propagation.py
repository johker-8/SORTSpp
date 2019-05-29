#propagation test

import scipy
import numpy as np
import matplotlib.pyplot as plt

import dpt_tools as dpt
import TLE_tools as tle
import propagator_sgp4

prop = propagator_sgp4.PropagatorSGP4(
    polar_motion = False,
    out_frame = 'ITRF',
)

ut0 = 1241136000.0

# modified Julian day epoch
jd0 = dpt.unix_to_jd(ut0)
mjd0 = dpt.jd_to_mjd(jd0)
R_e = 6371.0
m_to_A = 128.651
mass = 0.8111E+04

init_data = {
    'a': 7159.5*1e3,
    'e': 0.0001,
    'inc': 98.55,
    'raan': 248.99,
    'aop': 90.72,
    'mu0': 47.37,
    'mjd0': mjd0,
    'C_D': 2.3,
    'C_R': 1.0,
    'm': mass,
    'A': mass/m_to_A,
}

t = np.linspace(0, 3600*24, num = 1000, dtype=np.float64)

ecefs_1 = prop.get_orbit(t, **init_data)

ind = int(len(t)//2)

state0 = ecefs_1[:, ind]
state0 = tle.ITRF_to_TEME(state0, jd0 + t[ind]/(3600.0*24.0), 0.0, 0.0)

x, y, z, vx, vy, vz = state0
t_2 = t[ind:] - t[ind]

mjd1 = mjd0 + t[ind]/(3600.0*24.0)

ecefs_2 = prop.get_orbit_cart(t_2, x, y, z, vx, vy, vz, mjd1,
    m = init_data['m'],
    C_D = init_data['C_D'],
    C_R = init_data['C_R'],
    A = init_data['A'],
)

dr = np.linalg.norm(ecefs_1[:3, ind:] - ecefs_2[:3, :], axis=0)
dv = np.linalg.norm(ecefs_1[3:, ind:] - ecefs_2[3:, :], axis=0)

plt.plot(t_2, dr)
plt.show()