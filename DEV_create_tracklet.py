import numpy as np
import h5py
import os
import scipy.constants

import radar_library as rlib
import dpt_tools as dpt
import population_library
import simulate_tracklet
import ccsds_write
import correlator

import matplotlib.pyplot as plt

radar = rlib.eiscat_uhf()
radar.set_FOV(30.0, 25.0)

#tle files for envisat in 2016-09-05 to 2016-09-07 from space-track.


TLEs = [
    ('1 27386U 02009A   16249.14961597  .00000004  00000-0  15306-4 0  9994',
    '2 27386  98.2759 299.6736 0001263  83.7600 276.3746 14.37874511760117'),
    ('1 27386U 02009A   16249.42796553  .00000002  00000-0  14411-4 0  9997',
    '2 27386  98.2759 299.9417 0001256  82.8173 277.3156 14.37874515760157'),
    ('1 27386U 02009A   16249.77590267  .00000010  00000-0  17337-4 0  9998',
    '2 27386  98.2757 300.2769 0001253  82.2763 277.8558 14.37874611760201'),
    ('1 27386U 02009A   16250.12384028  .00000006  00000-0  15974-4 0  9995',
    '2 27386  98.2755 300.6121 0001252  82.5872 277.5467 14.37874615760253'),
    ('1 27386U 02009A   16250.75012691  .00000017  00000-0  19645-4 0  9999',
    '2 27386  98.2753 301.2152 0001254  82.1013 278.0311 14.37874790760345'),
]

from propagator_neptune import PropagatorNeptune
from propagator_sgp4 import PropagatorSGP4
from propagator_orekit import PropagatorOrekit

pop = population_library.tle_snapshot(TLEs, 
    sgp4_propagation=True,
    #sgp4_propagation=False,
    #propagator = PropagatorNeptune,
    #propagator = PropagatorOrekit,
    #propagator_options = {
    #    'in_frame': 'TEME',
    #    'out_frame': 'ITRF',
    #},
)

#it seems to around 25m^2 area
d = np.sqrt(25.0*4/np.pi)
pop['d'] = d

measurement_file = './data/uhf_test_data/events/pass-1473150428660000.h5'
#ccsds_file = './data/uhf_test_data/events/2002-009A-1473150428.tdm'
ccsds_file = './data/uhf_test_data/events/2002-009A-2016-09-06_08:27:08.tdm'


obs_data = ccsds_write.read_ccsds(ccsds_file)
jd_obs = dpt.mjd_to_jd(dpt.npdt2mjd(obs_data['date']))


date_obs = obs_data['date']
sort_obs = np.argsort(date_obs)
date_obs = date_obs[sort_obs]
r_obs = obs_data['range'][sort_obs]*0.5
v_obs = obs_data['doppler_instantaneous'][sort_obs]

#print(v_obs)
#exit()

#TO DEBUG
#jd_obs = jd_obs[:3]
#date_obs = date_obs[:3]
#r_obs = r_obs[:3]


jd_sort = jd_obs.argsort()
jd_obs = jd_obs[jd_sort]

jd_det = jd_obs[0]

jd_pop = dpt.mjd_to_jd(pop['mjd0'])

pop_id = np.argmin(np.abs(jd_pop - jd_det))
print(pop_id)
dels = [ind for ind in range(len(pop)) if ind != pop_id]
pop.delete(dels)
obj = pop.get_object(0)

#print(obj)

jd_obj = dpt.mjd_to_jd(obj.mjd0)

#print('Day difference: {}'.format(jd_det- jd_obj))

t_obs = (jd_obs - jd_obj)*(3600.0*24.0)

#correct for light time
lt_correction = r_obs/scipy.constants.c*1e3
t_obs -= lt_correction

#print(lt_correction)

'''
states = obj.get_orbit(t_obs)
ax = dpt.orbit3D(states)
radar.draw3d(ax)
plt.show()
'''

meas, fnames, ecef_stdevs = simulate_tracklet.create_tracklet(
    obj,
    radar,
    t_obs,
    hdf5_out=True,
    ccsds_out=True,
    dname="./tests/tmp_test_data",
    noise=False,
)

out_h5 = fnames[0] + '.h5'
out_ccsds = fnames[0] + '.tdm'

print('FILES: ', fnames)

with h5py.File(out_h5,'r') as h_det:
    pass
    #h_det['m_range']
    #h_det['m_range_rate']


sim_data = ccsds_write.read_ccsds(out_ccsds)

date_sim = sim_data['date']
sort_sim = np.argsort(date_sim)
date_sim = date_sim[sort_sim]

r_sim = sim_data['range'][sort_sim]*0.5
v_sim = sim_data['doppler_instantaneous'][sort_sim]*0.5

#lt_correction = np.round(r_sim/scipy.constants.c*1e6).astype(np.int64).astype('timedelta64[us]')

#date_sim_cor = date_sim + lt_correction

t_sim = dpt.jd_to_unix(dpt.mjd_to_jd(dpt.npdt2mjd(date_sim)))

'''
for ind in range(len(date_sim)):
    print('DIFF MJD {}: SIM {} - OBS {} = {} s'.format(
        ind,
        dpt.npdt2mjd(date_sim[ind]),
        dpt.npdt2mjd(date_obs[ind]),
        (dpt.npdt2mjd(date_sim[ind]) - dpt.npdt2mjd(date_obs[ind]))*3600.0*24.0,
    ))


print('r_obs: ',len(r_obs))
print('r_sim: ',len(r_sim))
'''
dat = {
    't': t_sim,
    'r': r_sim*1e3,
    'v': v_sim*1e3,
}

cdat = correlator.correlate(
    data = dat,
    station = radar._rx[0],
    population = pop,
    metric = correlator.residual_distribution_metric,
    n_closest = 1,
    out_file = None,
    verbose = False,
    MPI_on = False,
)

correlator.plot_correlation(dat, cdat[0])


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(211)
ax.plot(t_sim - t_sim[0], r_sim - r_obs, 'ob')
ax.set_ylabel('sim vs obs error [km]')
ax.set_xlabel('relative time [s]')
ax.set_title('Real TDM vs simulated TDM')

#print(v_sim)
#print(v_obs)

ax = fig.add_subplot(212)
ax.plot(t_sim - t_sim[0], v_sim - v_obs, 'ob')
ax.set_ylabel('sim vs obs error [km/s]')
ax.set_xlabel('relative time [s]')
ax.set_title('Real TDM vs simulated TDM')


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(211)
ax.plot(r_sim - r_obs, 'ob')
ax.set_ylabel('sim vs obs error [km]')
ax.set_xlabel('index')
ax.set_title('Real TDM vs simulated TDM')

ax = fig.add_subplot(212)
ax.plot(v_sim - v_obs, 'ob')
ax.set_ylabel('sim vs obs error [km/s]')
ax.set_xlabel('index')
ax.set_title('Real TDM vs simulated TDM')


plt.show()

os.remove(out_h5)
print('removed "{}"'.format(out_h5))

os.remove(out_ccsds)
print('removed "{}"'.format(out_ccsds))

sat_folder = os.sep.join(fnames[0].split(os.sep)[:-1])
os.rmdir(sat_folder)
print('removed "{}"'.format(sat_folder))
