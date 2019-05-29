import numpy as np
import scipy.constants as consts
import numpy.testing as nt
import TLE_tools as tle
import dpt_tools as dpt
from propagator_orekit import PropagatorOrekit
import matplotlib.pyplot as plt


mjd0 = dpt.jd_to_mjd(2457126.2729)

R_E = 6353.0e3

a = R_E*2.0
orb_init_list = []

orb_init_list.append(np.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
orb_init_list.append(np.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 270], dtype=np.float))
orb_init_list.append(np.array([R_E*1.2, 1e-9, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 0.0, 0.0, 0.0], dtype=np.float))
orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 120.0, 0.0, 0.0], dtype=np.float))
orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 0.0, 35.0, 0.0], dtype=np.float))
orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 120.0, 0.0, 0.0], dtype=np.float))
orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 0.0, 35.0, 0.0], dtype=np.float))
orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 120.0, 35.0, 0.0], dtype=np.float))

np.random.seed(12398774)

orb_range = np.array([a, 0.9, 180, 360, 360, 360], dtype=np.float)
orb_offset = np.array([R_E*1.1, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float)
while len(orb_init_list) < 5000:
    orb = np.random.rand(6)
    orb = orb_offset + orb*orb_range
    if orb[0]*(1.0 - orb[1]) > R_E+200e3:
        orb_init_list.append(orb)

np.random.seed(None)

orbs_pass = []
orbs = []
cols = []
orbs_fail = []
fail_inds = []
fail_err = []
fail_errv = []

prop = PropagatorOrekit(
        in_frame='EME',
        out_frame='EME',
)

for ind, kep in enumerate(orb_init_list):

    t = np.array([0.0])

    state_ref = dpt.kep2cart(kep, m=1.0, M_cent=prop.M_earth, radians=False)

    '''
    state = prop.get_orbit(
            t=t, mjd0=mjd0,
            a=kep[0], e=kep[1], inc=kep[2],
            raan=kep[4], aop=kep[3], mu0=dpt.true2mean(kep[5], kep[1], radians=False),
            C_D=2.3, m=1.0, A=1.0, C_R=1.0,
            radians=False,
        )
    '''
    state = prop.get_orbit_cart(
                t=t, mjd0=mjd0,
                x=state_ref[0], y=state_ref[1], z=state_ref[2],
                vx=state_ref[3], vy=state_ref[4], vz=state_ref[5],
                C_D=2.3, m=1.0, A=1.0, C_R=1.0,
            )
    state_diff= np.abs(state_ref - state[:,0])
    
    try:
        nt.assert_array_less(state_diff[:3], np.full((3,), 1e-3, dtype=state_diff.dtype))
        nt.assert_array_less(state_diff[3:], np.full((3,), 1e-5, dtype=state_diff.dtype))
        orbs_pass.append(kep.tolist())
        cols.append(0.0)
    except AssertionError as err:
        fail_inds.append(ind)
        orbs_fail.append(kep.tolist())
        cols.append(1.0)

    err = np.linalg.norm(state_diff[:3])
    errv = np.linalg.norm(state_diff[3:])
    
    fail_err.append(err)
    fail_errv.append(errv)


    orbs.append(kep.tolist())


orbs = np.array(orbs)
cols = np.array(cols)
fail_err = np.array(fail_err)
fail_errv = np.array(fail_errv)
orbs_fail = np.array(orbs_fail)
orbs_pass = np.array(orbs_pass)

print('FAIL / TOTAL: {} / {}'.format(len(fail_inds), len(orb_init_list)))

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(211)
ax.hist(fail_err, 100)
ax.set_xlabel('Position error')
ax = fig.add_subplot(212)
ax.hist(fail_errv, 100)
ax.set_xlabel('Velocity error')


plt.show()

dpt.orbits(orbs[fail_err > 1.0,:], **{'title': 'State recovery error: Color scale $\log_{10}(r_{error} [m])$', 'unit': 'm', 'color': np.log(fail_err[fail_err > 1.0])})
#dpt.orbits(orbs, **{'title': 'State recovery error: Color scale $\log_{10}(r_{error} [m])$', 'unit': 'm', 'color': np.log(fail_err)})
#dpt.orbits(orbs, **{'title': 'State recovery error: Color scale $\log_{10}(v_{error} [m/s])$', 'unit': 'm', 'color': np.log(fail_errv)})
