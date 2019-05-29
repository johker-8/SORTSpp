import numpy as np
import scipy.constants as consts
import numpy.testing as nt
import TLE_tools as tle
import dpt_tools as dpt
import propagator_sgp4
import matplotlib.pyplot as plt



'''

mjd0 = dpt.jd_to_mjd(2457126.2729)
a = 6800e3

orb_init = np.array([a*1.2, 0.1, 45.0, 0.0, 24.0, 0.0], dtype=np.float)

M_earth = propagator_sgp4.SGP4.GM*1e9/consts.G

state_TEME = dpt.kep2cart(orb_init, m=0.0, M_cent=M_earth, radians=False)*1e-3

mean_elements = tle.TEME_to_TLE_dev_version(state_TEME, kepler=False)

state = propagator_sgp4.sgp4_propagation(mjd0, mean_elements, B=0, dt=0.0)

state_diff = np.abs(state - state_TEME)*1e3

print(state_diff)



raise Exception('tes')

'''


mjd0 = dpt.jd_to_mjd(2457126.2729)
ar = 1.7
aof = 1.8
#aof = 1.1
orb_init_list = []

orb_range = np.array([6353.0e3*ar, 0.5, 180, 360, 360, 360], dtype=np.float)
orb_offset = np.array([6353.0e3*aof, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float)
test_n = 10000
while len(orb_init_list) < test_n:
    orb = np.random.rand(6)
    orb = orb_offset + orb*orb_range
    if orb[0]*(1.0 - orb[1]) > 6600e3:
        orb_init_list.append(orb)
#orb_init_list.append(np.array([a, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
#orb_init_list.append(np.array([a, 0, 0.0, 0.0, 0.0, 270], dtype=np.float))
#orb_init_list.append(np.array([a, 1e-9, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
#orb_init_list.append(np.array([a, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
#orb_init_list.append(np.array([a, 0.2, 75.0, 0.0, 0.0, 0.0], dtype=np.float))
#orb_init_list.append(np.array([a, 0.2, 0.0, 120.0, 0.0, 0.0], dtype=np.float))
#orb_init_list.append(np.array([a, 0.2, 0.0, 0.0, 35.0, 0.0], dtype=np.float))
#orb_init_list.append(np.array([a, 0.2, 75.0, 120.0, 0.0, 0.0], dtype=np.float))
#orb_init_list.append(np.array([a, 0.2, 75.0, 0.0, 35.0, 0.0], dtype=np.float))
#orb_init_list.append(np.array([a, 0.2, 75.0, 120.0, 35.0, 0.0], dtype=np.float))

orbs_pass = []
orbs = []
mean_orbs = []
cols = []
orbs_fail = []
fail_inds = []
fail_err = []
fail_errv = []
for ind, kep in enumerate(orb_init_list):
    #print('{}/{} done'.format(ind, len(orb_init_list)))
    M_earth = propagator_sgp4.SGP4.GM*1e9/consts.G

    state_TEME = dpt.kep2cart(kep, m=0.0, M_cent=M_earth, radians=False)*1e-3

    mean_elements = tle.TEME_to_TLE(state_TEME, mjd0=mjd0, kepler=False)
    mlst = mean_elements.tolist()
    mlst[0] *= 1e3
    mlst[4] = mean_elements[3]
    mlst[3] = mean_elements[4]
    mlst[5] = dpt.mean2true(mean_elements[5], mean_elements[1], radians=True)
    mean_orbs.append(mlst)
    state = propagator_sgp4.sgp4_propagation(mjd0, mean_elements, B=0, dt=0.0)

    state_diff = np.abs(state - state_TEME)*1e3

    try:
        nt.assert_array_less(state_diff[:3], np.full((3,), 100, dtype=state_diff.dtype))
        nt.assert_array_less(state_diff[3:], np.full((3,), 10, dtype=state_diff.dtype))
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


mean_orbs = np.array(mean_orbs)
mean_orbs[:,2:] = np.degrees(mean_orbs[:,2:])
orbs = np.array(orbs)
cols = np.array(cols)
fail_err = np.array(fail_err)
fail_errv = np.array(fail_errv)
orbs_fail = np.array(orbs_fail)
orbs_pass = np.array(orbs_pass)

print('FAIL / TOTAL: {} / {}'.format(len(fail_inds), len(orb_init_list)))
print('NAN  / TOTAL: {} / {}'.format(len(orb_init_list) - len(fail_err), len(orb_init_list)))

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(211)
ax.hist(fail_err, 100)
ax.set_xlabel('Position error')
ax = fig.add_subplot(212)
ax.hist(fail_errv, 100)
ax.set_xlabel('Velocity error')

fig = plt.figure(figsize=(15,15))

ax = fig.add_subplot(211)
ax.hist(fail_err[fail_err < 1e3], 100)
ax.set_xlabel('Position error')
ax = fig.add_subplot(212)
ax.hist(fail_errv[fail_errv < 0.1e3], 100)
ax.set_xlabel('Velocity error')

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.semilogy(orbs[:,0]/6353.0e3, fail_err, '.b', alpha=0.25)
ax.set_ylabel('position error [m]')
ax.set_xlabel('Semi-major axis [R_E]')

plt.show()


dpt.orbits(mean_orbs, **{'title': 'Mean orbits', 'unit': 'm'})
#dpt.orbits(orbs, **{'title': 'Orbits test TEME to TLE', 'unit': 'm', 'color': cols})
dpt.orbits(orbs, **{'title': 'Test TEME to TLE error: Color scale $\log_{10}(r_{error} [m])$', 'unit': 'm', 'color': np.log(fail_err)})
dpt.orbits(orbs[fail_err < 1.0,:], **{'title': 'Test TEME to TLE error: Color scale $\log_{10}(r_{error} [m])$', 'unit': 'm', 'color': np.log(fail_err[fail_err < 1.0])})
dpt.orbits(orbs[fail_err > 1.0,:], **{'title': 'Test TEME to TLE error: Color scale $\log_{10}(r_{error} [m])$', 'unit': 'm', 'color': np.log(fail_err[fail_err > 1.0])})

plt.show()



#if orbs_fail.shape[0] > 0:
#    dpt.orbits(orbs_fail, **{'show': False, 'title': 'fail', 'unit': 'm'})
#if orbs_pass.shape[0] > 0:
#    dpt.orbits(orbs_pass, **{'show': False, 'title': 'PASS', 'unit': 'm'})
#plt.show()

'''
reci = np.array([
        -5339.76186573000,
        5721.43584226500,
        921.276953805000,
    ], dtype=np.float)

veci = np.array([
        -4.88969089550000,
        -3.83304653050000,
        3.18013811100000,
    ], dtype=np.float)

rad = -7000.0
circ = np.linspace(0,np.pi*2,num=100,dtype=np.float)

orbs = np.empty((6,100), dtype=np.float)

for ind,th in enumerate(circ):
    reci0 = np.array([
            np.cos(th),
            np.sin(th),
            0,
        ], dtype=np.float)*rad
    veci0 = np.array([
            -np.sin(th),
            np.cos(th),
            0,
        ], dtype=np.float)*np.linalg.norm(veci)

    a1, eo, xno, xmo, xincl, omegao, xnodeo = tle.ECI_to_TLE(reci0, veci0)
    orbs[:,ind] = tle.TLE_to_mean_elements(a1, eo, xno, xmo, xincl, omegao, xnodeo)

#dpt.orbits(orbs.T, {})


mjd0 = dpt.jd_to_mjd(2457126.2729)



state_TEME = np.concatenate((reci*1e3, veci*1e3), axis=0)

M_earth = propagator_sgp4.SGP4.GM*1e9/consts.G

kep = dpt.cart2kep(state_TEME, m=0.0, M_cent=M_earth, radians=False)
kep[0] *= 1e-3
kep[5] = dpt.true2mean(kep[5], kep[1], radians=False)

a1, eo, xno, xmo, xincl, omegao, xnodeo = tle.ECI_to_TLE(reci, veci)
'''
'''
        a0     = mean_elements[0]         # Semi-major (a') at epoch [km]
        e0     = mean_elements[1]         # Eccentricity at epoch
        i0     = mean_elements[2]         # Inclination at epoch
        raan0  = mean_elements[3]         # RA of the ascending node at epoch
        aop0   = mean_elements[4]         # Argument of perigee at epoch
        M0     = mean_elements[5]         # Mean anomaly at epoch

    output:
     * a1     = mean semi major axis (kilometers)
     * eo     = orbital eccentricity (non-dimensional)
     * xno    = mean motion (orbits per day)
     * xmo    = mean anomaly (radians)
     * xincl  = orbital inclination (radians)
     * omegao = argument of perigee (radians)
     * xnodeo = right ascension of ascending node (radians)


'''
'''
labels = ['Semi-major axis: {} km', 'Eccentricity: {}', 'Inclination: {} deg', 'Argument of perigee: {} deg', 'RA of ascending node: {} deg', 'Mean anomaly: {} deg']

print('--------- Instantaneous Orbital elements:')
for ind, lab in enumerate(labels):
    print(lab.format(kep[ind]))


print('--------- Mean Orbital elements:')
print(labels[0].format(a1))
print(labels[1].format(eo))
print(labels[2].format(np.degrees(xincl)))
print(labels[3].format(np.degrees(omegao)))
print(labels[4].format(np.degrees(xnodeo)))
print(labels[5].format(np.degrees(xmo)))

print('--------- Difference Orbital elements:')
print(labels[0].format(kep[0] - a1))
print(labels[1].format(kep[1] - eo))
print(labels[2].format(kep[2] - np.degrees(xincl)))
print(labels[3].format(kep[3] - np.degrees(omegao)))
print(labels[4].format(kep[4] - np.degrees(xnodeo)))
print(labels[5].format(kep[5] - np.degrees(xmo)))

print('------------ ## ------------')
mean_elements = tle.TLE_to_mean_elements(a1, eo, xno, xmo, xincl, omegao, xnodeo)

state = propagator_sgp4.sgp4_propagation(mjd0, mean_elements, B=0, dt=0.0)*1e3

state_diff_correct = state - state_TEME

print('Mean Element calculated state: ', state)
print('True state                   : ', state_TEME)
print('State difference             : ', state_diff_correct)
print('Position error: {} m'.format(np.linalg.norm(state_diff_correct[:3])))
print('Velocity error: {} m/s'.format(np.linalg.norm(state_diff_correct[3:])))


inst_elements = kep.copy()
inst_elements[2:] = np.radians(inst_elements[2:])
tmp = inst_elements[3]
inst_elements[3] = inst_elements[4]
inst_elements[4] = tmp
del tmp

print('------------ ## ------------')
print('Mean vs inst element difference: ', mean_elements-inst_elements)
print('------------ ## ------------')

state_direct = propagator_sgp4.sgp4_propagation(mjd0, inst_elements, B=0, dt=0.0)*1e3

state_diff = state_direct - state_TEME

print('Mean Element calculated state: ', state_direct)
print('True state                   : ', state_TEME)
print('State difference             : ', state_diff)
print('Position error: {} m'.format(np.linalg.norm(state_diff[:3])))
print('Velocity error: {} m/s'.format(np.linalg.norm(state_diff[3:])))
print('------------ ## ------------')

'''