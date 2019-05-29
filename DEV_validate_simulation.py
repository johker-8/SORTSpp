#!/usr/bin/env python

import time
import sys
import os
import logging
import glob
import shutil
import pymc3 as pm

import h5py
import scipy
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# SORTS imports CORE
import population_library as plib
from simulation import Simulation
from population import Population

#SORTS Libraries
import radar_library as rlib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib
import plothelp
import dpt_tools as dpt
import ccsds_write

#from propagator_neptune import PropagatorNeptune
from propagator_sgp4 import PropagatorSGP4
#from propagator_orekit import PropagatorOrekit

def gen_pop():

    mass=0.8111E+04
    diam=0.8960E+01
    m_to_A=128.651
    a=7159.5
    e=0.0001
    i=98.55
    raan=248.99
    aop=90.72
    M=47.37
    A=mass/m_to_A
    # epoch in unix seconds
    ut0=1241136000.0
    # modified julian day epoch
    mjd0 = dpt.jd_to_mjd(dpt.unix_to_jd(ut0))

    print('\n' + 'mjd0: {}'.format(mjd0) + '\n')

    _params = ['A', 'm', 'd', 'C_D', 'C_R']

    pop = Population(
        name='ENVISAT',
        extra_columns = _params,
        space_object_uses = [True]*len(_params),
        #propagator = PropagatorNeptune,
        propagator = PropagatorSGP4,
        propagator_options = {
            'out_frame': 'ITRF'
        },
    )

    pop.allocate(1)

    pop['oid'][0] = 1
    pop['a'][0] = a
    pop['e'][0] = e
    pop['i'][0] = i
    pop['raan'][0] = raan
    pop['aop'][0] = aop
    pop['mu0'][0] = M
    pop['mjd0'][0] = mjd0
    pop['A'][0] = A
    pop['m'][0] = mass
    pop['d'][0] = 1.0
    pop['C_D'][0] = 2.3
    pop['C_R'][0] = 1.0

    return pop

def validate_simulation(root):

    sim_root = root + '/tests/tmp_test_data/ENVISAT_TRACKING'

    SIM_TIME = 15.0

    pop = gen_pop()

    #initialize the radar setup
    radar = rlib.eiscat_3d(beam='interp', stage=1)

    radar.set_FOV(max_on_axis=90.0, horizon_elevation=30.0)
    radar.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=1.0)
    radar.set_TX_bandwith(bw = 1.0e6)

    sim = Simulation(
        radar = radar,
        population = pop,
        root = sim_root,
        scheduler = schlib.dynamic_scheduler,
        simulation_name = 'ENVISAT validation',
    )

    sim.set_log_level(logging.INFO)

    sim.observation_parameters(
        duty_cycle=0.25,
        SST_fraction=1.0,
        tracking_fraction=1.0,
        SST_time_slice=0.2,
    )
    sim.simulation_parameters(
        tracklet_noise=True,
        max_dpos=50e3,
        auto_synchronize=True,
    )

    #We know all!
    sim.catalogue.maintain(slice(None))

    ################## RUNNING #####################

    sim.run_observation(SIM_TIME)

    sim.status(fout='{}h_to_{}h'.format(0, int(SIM_TIME)))

    sim.print_maintenance()
    sim.print_detections()

    sim.set_scheduler_args(
        reward_function = rflib.rewardf_exp_peak_SNR_tracklet_len,
        reward_function_config = {
            'sigma_t': 60.0*5.0,
            'lambda_N': 50.0,
        },
        logger = sim.logger,
    )

    sim.run_scheduler()
    sim.print_tracks()
    sim.print_tracklets()

    sim.status(fout='{}h_to_{}h_scheduler'.format(0, int(SIM_TIME)))

    sim.generate_tracklets()
    sim.generate_priors()


def test_prior():
    import plothelp
    #initialize the radar setup
    radar = rlib.eiscat_3d(beam='interp', stage=1)

    radar.set_FOV(max_on_axis=90.0, horizon_elevation=10.0)
    radar.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=1.0)
    radar.set_TX_bandwith(bw = 1.0e6)

    tx = radar._tx[0]

    pop = gen_pop()
    obj = pop.get_object(0)

    ecefs = obj.get_orbit(np.linspace(0,2*3600,num=2000, dtype=np.float64))

    fname = '/home/danielk/IRF/IRF_GITLAB/SORTSpp/tests/tmp_test_data/ENVISAT_TRACKING/master/prior/1_init.oem'
    '''
    fname_tracklet = '/home/danielk/IRF/IRF_GITLAB/SORTSpp/tests/tmp_test_data/ENVISAT_TRACKING/master/tracklets/1/track-1241140623-1-0_0.tdm'

    obs_data = ccsds_write.read_ccsds(fname_tracklet)
    sort_obs = np.argsort(obs_data['date'])
    obs_data = obs_data[sort_obs]

    r_sim = obs_data['range']*0.5
    v_sim = obs_data['doppler_instantaneous']*0.5
    t_sim = (obs_data['date'] - obs_data['date'][0])/np.timedelta64(1, 's')
    '''

    data, meta = ccsds_write.read_oem(fname)
    print(data)
    print('== META ==')
    for key, val in meta.items():
        print('{}: {}'.format(key,val))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)

    ax.plot(data['x'], data['y'], data['z'], '-b', label = 'Prior', alpha = 0.75)
    ax.plot(ecefs[0,:], ecefs[1,:], ecefs[2,:], '-g', label = 'Propagation', alpha = 0.75)
    ax.plot([tx.ecef[0]], [tx.ecef[1]], [tx.ecef[2]], 'or', label = 'E3D')
    ax.legend()

    plt.show()


def test_OD(root, sub_path):

    import orbit_determination
    import TLE_tools as tle
    import dpt_tools as dpt
    import radar_library as rlib
    import propagator_sgp4
    #import propagator_orekit
    #import propagator_neptune
    import ccsds_write

    radar = rlib.eiscat_3d(beam='interp', stage=1)

    radar.set_FOV(max_on_axis=90.0, horizon_elevation=10.0)
    radar.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=1.0)
    radar.set_TX_bandwith(bw = 1.0e6)
    
    #prop = propagator_neptune.PropagatorNeptune()
    prop = propagator_sgp4.PropagatorSGP4()

    mass=0.8111E+04
    diam=0.8960E+01
    m_to_A=128.651

    params = dict(
        A = {
            'dist': None,
            'val': mass/m_to_A,
        },
        d = {
            'dist': None,
            'val': diam,
        },
        m = {
            'dist': None,
            'val': mass,
        },
        C_D = {
            'dist': None,
            'val': 2.3,
        },
    )

    fname = glob.glob(root + sub_path + '*.oem')[0]
    prior_data, prior_meta = ccsds_write.read_oem(fname)
    prior_sort = np.argsort(prior_data['date'])
    prior_data = prior_data[prior_sort][0]

    prior_mjd = dpt.npdt2mjd(prior_data['date'])
    prior_jd = dpt.mjd_to_jd(prior_mjd)

    state0 = np.empty((6,), dtype=np.float64)
    state0[0] = prior_data['x']
    state0[1] = prior_data['y']
    state0[2] = prior_data['z']
    state0[3] = prior_data['vx']
    state0[4] = prior_data['vy']
    state0[5] = prior_data['vz']
    
    #state0_ITRF = state0.copy()
    state0 = tle.ITRF_to_TEME(state0, prior_jd, 0.0, 0.0)
    #state0_TEME = state0.copy()
    
    #state0_ITRF_ref = tle.TEME_to_ITRF(state0_TEME, prior_jd, 0.0, 0.0)
    
    #print(state0_ITRF_ref - state0_ITRF)
    #exit()
    
    data_folder = root + sub_path
    
    data_h5 = glob.glob(data_folder + '*.h5')
    
    data_h5_sort = np.argsort(np.array([int(_h.split('/')[-1].split('-')[1]) for _h in data_h5])).tolist()
    
    true_prior_h5 = data_h5[data_h5_sort[0]]
    true_obs_h5 = data_h5[data_h5_sort[1]]
    
    print(true_prior_h5)
    print(true_obs_h5)
    
    with h5py.File(true_prior_h5, 'r') as hf:
        true_prior = hf['true_state'].value.T*1e3
        true_prior_jd = dpt.unix_to_jd(hf['true_time'].value)
    
    print('-- True time diff prior [s] --')
    prior_match_ind = np.argmin(np.abs(true_prior_jd-prior_jd))
    jd_diff = prior_jd - true_prior_jd[prior_match_ind]
    state0_true = true_prior[:,prior_match_ind]
    state0_true = tle.ITRF_to_TEME(state0_true, true_prior_jd[prior_match_ind], 0.0, 0.0)
    
    print(prior_match_ind)
    print(jd_diff*3600.0*24.0)

    with h5py.File(true_obs_h5, 'r') as hf:
        true_obs = hf['true_state'].value.T*1e3
        true_obs_jd = dpt.unix_to_jd(hf['true_time'].value)

    data_tdm = glob.glob(data_folder + '*.tdm')
    #this next line i wtf, maybe clean up
    data_tdm_sort = np.argsort(np.array([int(_h.split('/')[-1].split('-')[-1][2]) for _h in data_tdm])).tolist()
    
    ccsds_files = [data_tdm[_tdm] for _tdm in data_tdm_sort]
    
    print('prior true vs prior mean')
    print(state0_true - state0)
    
    for _fh in ccsds_files:
        print(_fh)
    
    r_obs_v = []
    r_sig_v = []
    v_obs_v = []
    v_sig_v = []
    t_obs_v = []

    for ccsds_file in ccsds_files:
        obs_data = ccsds_write.read_ccsds(ccsds_file)
        sort_obs = np.argsort(obs_data['date'])
        obs_data = obs_data[sort_obs]
        jd_obs = dpt.mjd_to_jd(dpt.npdt2mjd(obs_data['date']))

        date_obs = obs_data['date']
        sort_obs = np.argsort(date_obs)
        date_obs = date_obs[sort_obs]
        r_obs = obs_data['range'][sort_obs]*1e3 #to m
        v_obs = -obs_data['doppler_instantaneous'][sort_obs]*1e3 #to m/s
        #v_obs = obs_data['doppler_instantaneous'][sort_obs]*1e3 #to m/s
        r_sig = 2.0*obs_data['range_err'][sort_obs]*1e3 #to m
        v_sig = 2.0*obs_data['doppler_instantaneous_err'][sort_obs]*1e3 #to m/s

        #TRUNCATE FOR DEBUG
        inds = np.linspace(0,len(jd_obs)-1,num=10,dtype=np.int64)
        jd_obs = jd_obs[inds]
        r_obs = r_obs[inds]
        v_obs = v_obs[inds]
        r_sig = r_sig[inds]
        v_sig = v_sig[inds]

        if ccsds_file.split('/')[-1].split('.')[0] == true_obs_h5.split('/')[-1].split('.')[0]:
            print('-- True time diff obs [s] --')
            jd_diff = jd_obs - true_obs_jd[inds]
            print(jd_diff*3600.0*24.0)

        #r_sig = np.full(r_obs.shape, 100.0, dtype=r_obs.dtype)
        #v_sig = np.full(v_obs.shape, 10.0, dtype=v_obs.dtype)
        
        r_obs_v.append(r_obs)
        r_sig_v.append(r_sig)
        v_obs_v.append(v_obs)
        v_sig_v.append(v_sig)

        t_obs = (jd_obs - prior_jd)*(3600.0*24.0)

        #correct for light time approximently
        lt_correction = r_obs*0.5/scipy.constants.c
        t_obs -= lt_correction

        t_obs_v.append(t_obs)


    print('='*10 + 'Dates' + '='*10)
    print('{:<8}: {} JD'.format('Prior', prior_jd))
    for ind, _jd in enumerate(jd_obs):
        print('Obs {:<4}: {} JD'.format(ind, _jd))
    

    print('='*10 + 'Observations' + '='*10)
    print(len(jd_obs))
    
    prior = {}
    prior['cov'] = np.diag([1e3, 1e3, 1e3, 1e1, 1e1, 1e1])*1.0
    prior['mu'] = state0
    
    print('='*10 + 'Prior Mean' + '='*10)
    print(prior['mu'])

    print('='*10 + 'Prior Covariance' + '='*10)
    print(prior['cov'])
    
    rx_ecef = []
    for rx in radar._rx:
        rx_ecef.append(rx.ecef)
    tx_ecef = radar._tx[0].ecef
    tune = 0

    trace = orbit_determination.determine_orbit(
        num = 2000,
        r = r_obs_v,
        sd_r = r_sig_v,
        v = v_obs_v,
        sd_v = v_sig_v,
        grad_dx = [10.0]*3 + [1.0]*3,
        rx_ecef = rx_ecef,
        tx_ecef = tx_ecef,
        t = t_obs_v,
        mjd0 = prior_mjd,
        params = params,
        prior = prior,
        propagator = prop,
        step = 'Metropolis',
        step_opts = {
            'scaling': 0.75,
        },
        pymc_opts = {
            'tune': tune,
            'discard_tuned_samples': True,
            'cores': 1,
            'chains': 1,
            'parallelize': True,
        },
    )
    
    #if comm.rank != 0:
    #    exit()

    var = ['$X$ [km]', '$Y$ [km]', '$Z$ [km]', '$V_X$ [km/s]', '$V_Y$ [km/s]', '$V_Z$ [km/s]']

    fig = plt.figure(figsize=(15,15))

    for ind in range(6):
        ax = fig.add_subplot(231+ind)
        ax.plot(trace['state'][:,ind]*1e-3)
        ax.set(
            xlabel='Iteration',
            ylabel='{}'.format(var[ind]),
        )


    state1 = np.mean(trace['state'], axis=0)

    print('='*10 + 'Trace summary' + '='*10)
    print(pm.summary(trace))

    _form = '{:<10}: {}'

    print('='*10 + 'Prior Mean' + '='*10)
    for ind in range(6):
        print(_form.format(var[ind], state0[ind]*1e-3))

    print('='*10 + 'Posterior state mean' + '='*10)
    for ind in range(6):
        print(_form.format(var[ind], state1[ind]*1e-3))
    
    stated = state1 - state0

    print('='*10 + 'State shift' + '='*10)
    for ind in range(6):
        print(_form.format(var[ind], stated[ind]*1e-3))

    print('='*10 + 'True posterior' + '='*10)
    for ind in range(6):
        print(_form.format(var[ind], state0_true[ind]*1e-3))
    
    print('='*10 + 'Posterior error' + '='*10)
    for ind in range(6):
        print(_form.format(var[ind],(state1[ind] - state0_true[ind])*1e-3))
    
    print('='*10 + 'Parameter shift' + '='*10)
    theta0 = {}
    theta1 = {}
    for key, val in params.items():
        if val['dist'] is not None:
            theta0[key] = val['mu']
            theta1[key] = np.mean(trace[key], axis=0)[0]
            print('{}: {}'.format(key, theta1[key] - theta0[key]))
        else:
            theta0[key] = val['val']
            theta1[key] = val['val']


    range_v_prior = []
    vel_v_prior = []
    range_v = []
    vel_v = []
    range_v_true = []
    vel_v_true = []

    for rxi in range(len(rx_ecef)):
        t_obs = t_obs_v[rxi]
        print('Generating tracklet simulated data RX {}: {} points'.format(rxi, len(t_obs)))

        states0 = orbit_determination.propagate_state(state0, t_obs, dpt.jd_to_mjd(prior_jd), prop, theta0)
        states1 = orbit_determination.propagate_state(state1, t_obs, dpt.jd_to_mjd(prior_jd), prop, theta1)
        states0_true = orbit_determination.propagate_state(state0_true, t_obs, dpt.jd_to_mjd(prior_jd), prop, theta1)

        range_v_prior += [np.empty((len(t_obs), ), dtype=np.float64)]
        vel_v_prior += [np.empty((len(t_obs), ), dtype=np.float64)]
    
        range_v += [np.empty((len(t_obs), ), dtype=np.float64)]
        vel_v += [np.empty((len(t_obs), ), dtype=np.float64)]
    
        range_v_true += [np.empty((len(t_obs), ), dtype=np.float64)]
        vel_v_true += [np.empty((len(t_obs), ), dtype=np.float64)]

        for ind in range(len(t_obs)):
            range_v_prior[rxi][ind], vel_v_prior[rxi][ind] = orbit_determination.generate_measurements(states0[:,ind], rx_ecef[rxi], tx_ecef)
            range_v[rxi][ind], vel_v[rxi][ind] = orbit_determination.generate_measurements(states1[:,ind], rx_ecef[rxi], tx_ecef)
            range_v_true[rxi][ind], vel_v_true[rxi][ind] = orbit_determination.generate_measurements(states0_true[:, ind], rx_ecef[rxi], tx_ecef)


    prop_states = orbit_determination.propagate_state(
        state0,
        np.linspace(0, (np.max(jd_obs) - prior_jd)*(3600.0*24.0), num=1000),
        dpt.jd_to_mjd(prior_jd),
        prop,
        theta1,
    )
    
    '''
    pop = gen_pop()
    obj = pop.get_object(0)
    
    t_obs_pop = t_obs + (dpt.jd_to_mjd(prior_jd) - obj.mjd0)*3600.0*24.0
    states0_true2 = obj.get_state(t_obs_pop)
    
    print(states0_true2)
    print(states0_true2 - states0_true)
    '''
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    for ind, ecef in enumerate(rx_ecef):
        if ind == 0:
            ax.plot([ecef[0]], [ecef[1]], [ecef[2]], 'or', label='EISCAT 3D RX')
        else:
            ax.plot([ecef[0]], [ecef[1]], [ecef[2]], 'or')

    ax.plot(states0[0,:], states0[1,:], states0[2,:], 'xb', label = 'Prior', alpha = 0.75)
    ax.plot(states1[0,:], states1[1,:], states1[2,:], 'xr', label = 'Posterior', alpha = 0.75)
    ax.plot(prop_states[0,:], prop_states[1,:], prop_states[2,:], '-k', label = 'Prior-propagation', alpha = 0.5)
    ax.plot(true_prior[0,:], true_prior[1,:], true_prior[2,:], '-b', label = 'Prior-True', alpha = 0.75)
    ax.plot(true_obs[0,:], true_obs[1,:], true_obs[2,:], '-r', label = 'Posterior-True', alpha = 0.75)
    
    ax.legend()

    for rxi in range(len(rx_ecef)):

        fig = plt.figure(figsize=(15,15))
    
        t_obs_h = t_obs_v[rxi]/3600.0
        
        ax = fig.add_subplot(221)
        lns = []
        line1 = ax.plot(t_obs_h, (r_obs_v[rxi] - range_v[rxi])*1e-3, '-b', label='Maximum a posteriori: RX{}'.format(rxi))
        line0 = ax.plot(t_obs_h, (r_obs_v[rxi] - range_v_true[rxi])*1e-3, '.b', label='True prior: RX{}'.format(rxi))
        ax.set(
            xlabel='Time [h]',
            ylabel='2-way-Range residuals [km]',
        )
        ax2 = ax.twinx()
        line2 = ax2.plot(t_obs_h, (r_obs_v[rxi] - range_v_prior[rxi])*1e-3, '-k', label='Maximum a priori: RX{}'.format(rxi))
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='k')
    
        lns += line0+line1+line2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)
    
        ax = fig.add_subplot(222)
        lns = []
        line1 = ax.plot(t_obs_h, (v_obs_v[rxi] - vel_v[rxi])*1e-3, '-b', label='Maximum a posteriori: RX{}'.format(rxi))
        line0 = ax.plot(t_obs_h, (v_obs_v[rxi] - vel_v_true[rxi])*1e-3, '.b', label='True prior: RX{}'.format(rxi))
        ax.set(
            xlabel='Time [h]',
            ylabel='2-way-Velocity residuals [km/s]',
        )
        ax2 = ax.twinx()
        line2 = ax2.plot(t_obs_h, (v_obs_v[rxi] - vel_v_prior[rxi])*1e-3, '-k', label='Maximum a priori: RX{}'.format(rxi))
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='k')
        
        lns += line0+line1+line2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)

        ax = fig.add_subplot(223)
        ax.errorbar(t_obs_h, r_obs_v[rxi]*1e-3, yerr=r_sig_v[rxi]*1e-3, label='Measurements: RX{}'.format(rxi))
        ax.plot(t_obs_h, range_v[rxi]*1e-3, label='Maximum a posteriori: RX{}'.format(rxi))
        ax.plot(t_obs_h, range_v_prior[rxi]*1e-3, label='Maximum a priori: RX{}'.format(rxi))
        ax.set(
            xlabel='Time [h]',
            ylabel='2-way-Range [km]',
        )
        ax.legend()
        
        ax = fig.add_subplot(224)
        ax.errorbar(t_obs_h, v_obs_v[rxi]*1e-3, yerr=v_sig_v[rxi]*1e-3, label='Measurements: RX{}'.format(rxi))
        ax.plot(t_obs_h, vel_v[rxi]*1e-3, label='Maximum a posteriori: RX{}'.format(rxi))
        ax.plot(t_obs_h, vel_v_prior[rxi]*1e-3, label='Maximum a priori: RX{}'.format(rxi))
        ax.set(
            xlabel='Time [h]',
            ylabel='2-way-Velocity [km/s]',
        )
        ax.legend()
    
    #dpt.posterior(trace['state']*1e-3, var, show=False)
    plt.show()


if __name__=='__main__':
    root = '/home/waxarn333/repos/SORTSpp'
    sub_path = '/tests/tmp_test_data/ENVISAT_TRACKING/master'
    #sub_path = '/tests/tmp_test_data/ENVISAT_TRACKING/test_OD/'
    #sub_path = '/tests/tmp_test_data/ENVISAT_TRACKING/test_OD/prev/'
    validate_simulation(root)
    #test_prior()
    test_OD(root, sub_path)

    #shutil.rmtree(root)