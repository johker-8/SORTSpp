
import sys
import os
import time
import glob

import numpy as n
import scipy
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plothelp
import TLE_tools as tle
import dpt_tools as dpt
import space_object as so
from keplerian_sgp4 import sgp4_propagator as sgp_p
import radar_library as rlib


'''
TLE data from space-track.org
Radar data from EISCAT UHF

UHF location:
69.58649229 N 19.22592538 E, 85.554 m

'''

plot_data = False
print_data = False

radar = rlib.eiscat_uhf()

measurement_folder = './data/uhf_test_data/events'
tle_snapshot = './data/uhf_test_data/tle-201801.txt'

# measurement data
# ----------------
dets_list = glob.glob(measurement_folder + '/det-*.h5')

dets_data = []

for det in dets_list:
    with h5py.File(det,'r') as h_det:
        det_ID = int(h_det['idx'].value)
        


        r = h_det['r'].value*1e3
        t = h_det['t'].value
        v = h_det['v'].value

        t_sort = t.argsort()

        data_dict = {
            'r': r[t_sort],
            't': t[t_sort],
            'v': v[t_sort],
        }

    dets_data.append((det_ID, data_dict))

    if print_data:
        print('Detection - ID {}: {} tracklet points'.format(det_ID, data_dict['r'].size))
# ----------------

# TLE snapshot
# ----------------
tle_raw = [line.rstrip('\n') for line in open(tle_snapshot)]
if len(tle_raw) % 2 != 0:
    raise Exception('Not even number of lines [not TLE compatible]')


TLEs = zip(tle_raw[0::2], tle_raw[1::2])

'''
database = n.empty((len(TLEs), 8), dtype=n.float)
#id, epoch [JD], state

if plot_data:
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)
    plothelp.draw_earth_grid(ax)


ID = 0
object_list = []

for line1, line2 in TLEs:
    state_TEME, epoch = tle.TLE_to_TEME(line1,line2)
    
    PM_data = tle.get_Polar_Motion(epoch)

    state = tle.TEME_to_ITRF(state_TEME, epoch, PM_data[0,0], PM_data[0,1])

    database[ID,0] = ID
    database[ID,1] = epoch
    database[ID,2:] = state

    if plot_data:
        ax.plot([state[0]],[state[1]],[state[2]],".",alpha=0.5,color="black")

    if print_data:
        print('TLE - ID {} @ {}: {}'.format(ID, dpt.jd_to_date(epoch), state))

    ID += 1

if plot_data:
    plt.title("TLE snapshot")
    plt.show()

# ----------------
'''

# Detection correlation
#
# For each detection, propagate all objects to detection and correlate
# ----------------

n_closest = 5

loc_ecef = radar._tx[0].ecef.copy()
loc_ecef_norm = loc_ecef/n.linalg.norm(loc_ecef)

all_sat_ids = set([tle.tle_id(line1) for line1, line2 in TLEs])


for det_ID, data_dict in dets_data:

    t = data_dict['t']
    r = data_dict['r']
    v = data_dict['v']

    jd_check = dpt.unix_to_jd(t)

    r_ref = n.empty(r.shape, dtype=r.dtype)
    v_ref = n.empty(v.shape, dtype=v.dtype)

    correlation_data = {}
    #residulas = n.empty((4,1), dtype=n.float)

    t0 = time.time()

    jd_det = jd_check[0]

    PM_data = tle.get_Polar_Motion(jd_check)

    for sat_index, check_sat in enumerate(all_sat_ids):

        jd_tle_list = []
        jd_tle_index = []

        TLE_list_index = 0
        for line1, line2 in TLEs:
            sat_id = tle.tle_id(line1)
            if sat_id == check_sat:
                jd_tle_list.append(tle.tle_jd(line1))
                jd_tle_index.append(TLE_list_index)
            TLE_list_index += 1
        
        jd_diff = n.array([n.abs(x-jd_det) for x in jd_tle_list])

        TLE_list_index = jd_tle_index[n.argmin(jd_diff)]
        line1, line2 = TLEs[TLE_list_index]

        states_TEME = tle.TLE_propagation_TEME(line1, line2, jd_check)

        states = n.empty(states_TEME.shape, dtype=states_TEME.dtype)
        for jdi in range(jd_check.size):
            states[:,jdi] = tle.TEME_to_ITRF(states_TEME[:,jdi], jd_check[jdi], PM_data[jdi,0], PM_data[jdi,1])

            r_tmp = loc_ecef - states[:3,jdi]

            r_ref[jdi] = n.linalg.norm(r_tmp)
            v_ref[jdi] = n.dot(
                r_tmp/r_ref[jdi],
                states[3:,jdi],
            )

        residual_r_mu = n.mean(r_ref - r)
        residual_r_std = n.std(r_ref - r)
        residual_v_mu = n.mean(v_ref - v)
        residual_v_std = n.std(v_ref - v)
        residual_a = ((v_ref[-1] - v_ref[0]) - v[-1] - v[0])/(t[-1] - t[0])

        correlation_data[check_sat] = {
            'r_ref': r_ref.copy(),
            'v_ref': v_ref.copy(),
            'sat_id': check_sat,
            'sat_index': sat_index,
            'index': TLE_list_index,
            'stat': [residual_r_mu, residual_r_std, residual_v_mu, residual_v_std, residual_a]
        }

        print('Correlation Detection ID {}, sat {} of {} with ID {}'.format(
            det_ID,
            sat_index,
            len(all_sat_ids),
            check_sat,
        ))
        print('residual r: mu={} km,   sigma={} km'.format(
            residual_r_mu*1e-3,
            residual_r_std*1e-3,
        ))
        print('residual v: mu={} km/s, sigma={} km/s'.format(
            residual_v_mu*1e-3,
            residual_v_std*1e-3,
        ))
        print('residual a: {} km/s^2'.format(
            residual_a*1e-3,
        ))

        dt_time = (time.time() - t0)/60.0
        print('Time elapsed: {} min, time left: {} min'.format(
            dt_time,
            dt_time/float(sat_index+1)*(len(all_sat_ids) - float(sat_index)),
        ))

    match_metric = n.empty((len(correlation_data),), dtype = n.float)
    key_list = []
    key_cnt = 0
    for key, cdat in correlation_data.items():
        key_list.append(key)
        match_metric[key_cnt] = n.abs(cdat['stat'][0]) + n.abs(cdat['stat'][2]) +  n.abs(cdat['stat'][4])
        key_cnt += 1


    match = n.argmin(match_metric)
    all_match = n.argpartition(match_metric, n_closest)

    cdat = correlation_data[key_list[match]]

    r_ref = cdat['r_ref']
    v_ref = cdat['v_ref']

    h_corr = h5py.File(measurement_folder + '/' + str(det_ID) + '_corr.h5', 'w')

    for key, cdat in correlation_data.items():
        for dat_key, dat in cdat.items():
            h_corr[key+'/'+dat_key] = n.array(dat)

    h_corr.attrs['sat_match'] = key_list[match]
    h_corr['residuals'] = n.array(cdat['stat'])

    print('best match: SAT {}'.format(key_list[match]))
    print('43075?')
    print(cdat['stat'])


    for may_match in all_match[:n_closest]:

        cdat = correlation_data[key_list[may_match]]

        r_ref = cdat['r_ref']
        v_ref = cdat['v_ref']
        
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(211)
        ax.plot(t - t[0], r*1e-3, label='measurement')
        ax.plot(t - t[0], r_ref*1e-3, label='simulation')
        ax.set_ylabel('Range [km]')
        ax.set_xlabel('Time [s]')
        ax.set_title('Match {}'.format(key_list[may_match]))

        ax = fig.add_subplot(212)
        ax.plot(t - t[0], v*1e-3, label='measurement')
        ax.plot(t - t[0], v_ref*1e-3, label='simulation')
        ax.set_ylabel('Velocity [km/s]')
        ax.set_xlabel('Time [s]')
        ax.set_title('res a: {:.3f} m/s^2'.format(cdat['stat'][4]))

        plt.legend()

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(221)
        ax.hist((r_ref - r)*1e-3)
        ax.set_xlabel('Range residuals [km]')
        ax.set_title('res r: mu={:.3f} km, std={:.3f} km'.format(cdat['stat'][0]*1e-3, cdat['stat'][1]*1e-3))

        ax = fig.add_subplot(222)
        ax.hist((v_ref - v)*1e-3)
        ax.set_xlabel('Velocity residuals [km/s]')
        ax.set_title('res v: mu={:.3f} km/s, std={:.3f} km/s'.format(cdat['stat'][2]*1e-3, cdat['stat'][3]*1e-3))
        
        ax = fig.add_subplot(223)
        ax.plot(t - t[0], (r_ref - r)*1e-3)
        ax.set_ylabel('Range residuals [km]')
        ax.set_xlabel('Time [s]')

        ax = fig.add_subplot(224)
        ax.plot(t - t[0], (v_ref - v)*1e-3)
        ax.set_ylabel('Velocity residuals [km/s]')
        ax.set_xlabel('Time [s]')
    
    plt.show()