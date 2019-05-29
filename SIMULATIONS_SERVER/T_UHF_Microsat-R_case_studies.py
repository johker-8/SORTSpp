#!/usr/bin/env python
#
#
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import h5py
import sys
import os
import time

comm = MPI.COMM_WORLD

# replace this with the path to your sorts
sys.path.insert(0, "/home/danielk/PYTHON/SORTSpp")

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

import propagator_sgp4
import propagator_orekit
import dpt_tools as dpt
import coord

os.environ['TZ'] = 'GMT'
time.tzset()

####### RUN CONFIG #######
part = 3
campagin = 0
sim_v = 1
##########################


if campagin == 0:
    the_date = '_2019_04_02_'
    SIM_TIME = 24.0
    dt = np.datetime64('2019-04-02T12:01')
    mjd = dpt.npdt2mjd(dt)
    scan = rslib.beampark_model(
        lat = radar._tx[0].lat,
        lon = radar._tx[0].lon,
        alt = radar._tx[0].alt, 
        az = 90.0,
        el = 70.0,
    )
elif campagin == 1:
    the_date = '_2019_04_05_'
    SIM_TIME = 5.0
    dt = np.datetime64('2019-04-05T08:01')
    mjd = dpt.npdt2mjd(dt)
    scan = rslib.beampark_model(
        lat = radar._tx[0].lat,
        lon = radar._tx[0].lon,
        alt = radar._tx[0].alt, 
        az = 90.0,
        el = 45.0,
    )
if sim_v == 0:
    prop = 'SGP4'
    _prop = propagator_sgp4.PropagatorTLE
    _opts = {
        'out_frame': 'ITRF',
        'polar_motion': False,
    }
    branch_name = 'TLE_measurement' + the_date
    _label = 'TLE 44117-44173'

elif sim_v == 1:
    prop = 'Orekit'
    _prop = propagator_orekit.PropagatorOrekit
    _opts = {
        'in_frame': 'TEME',
        'out_frame': 'ITRF',
        'solar_activity_strength': 'WEAK',
    }
    branch_name = 'deb_simulation' + the_date
    _label = 'Simulation'

sim_root = '/ZFS_DATA/SORTSpp/FP_sims/T_UHF_MicrosatR_v2'


if comm.rank == 0:
    if not os.path.exists(sim_root):
        os.makedirs(sim_root)

#initialize the radar setup 
radar = rlib.eiscat_uhf()

radar.set_FOV(max_on_axis=25.0, horizon_elevation=50.0)
radar.set_SNR_limits(min_total_SNRdb=14.0, min_pair_SNRdb=14.0)

radar.set_scan(scan)

fname = sim_root + '/population_{}.h5'.format(branch_name)
if os.path.exists(fname):
    pop = Population.load(fname,
        propagator = _prop,
        propagator_options = _opts,
    )
else:

    if sim_v == 0:
        pop = plib.Microsat_R_debris_TLE(mjd)
        if comm.rank == 0:
            pop.save(fname)

    if sim_v == 1:
        if comm.rank == 0:
            pop = plib.simulate_Microsat_R_debris_v2(
                num = 1000,
                max_dv = 0.5e3,
                rho_range = [1000.0, 8000.0],
                mass_range = [0.005, 1.0],
                seed = 123546,
                propagator = _prop,
                propagator_options = _opts,
                mjd = mjd,
            )

            pop.save(fname)

        comm.barrier()
        if comm.rank != 0:
            pop = Population.load(fname,
                propagator = _prop,
                propagator_options = _opts,
            )

sim = Simulation(
    radar = radar,
    population = pop,
    root = sim_root,
    scheduler = schlib.dynamic_scheduler,
    simulation_name = 'EISCAT UHF scanning of MicrosatR'
)

sim.observation_parameters(
    duty_cycle=0.125,
    SST_fraction=1.0,
    tracking_fraction=0.0,
    SST_time_slice=0.2,
)
sim.simulation_parameters(
    tracklet_noise=True,
    max_dpos=50e3,
    auto_synchronize=True,
)


################## RUNNING #####################

if part == 1:
    sim.set_version(branch_name)
    sim.clear_simulation()
    sim.run_observation(SIM_TIME)

    sim.status(fout='{}h_to_{}h'.format(0, int(SIM_TIME)))

if part == 2:
    sim.set_version(branch_name)
    sim.load()

    sim.set_scheduler_args(
        logger = sim.logger,
    )

    sim.run_scheduler()

    sim.status(fout='{}h_to_{}h_scheduler'.format(0, int(SIM_TIME)))

if part == 3:
    sim.set_version(branch_name)
    sim.load()

    data_file = './data/microsatR_uhf.h5'

    with h5py.File(data_file, 'r') as hf:
        t_obs = hf['t'].value
        r_obs = hf['r'].value
        v_obs = hf['v'].value

    t_obs = (dpt.jd_to_mjd(dpt.unix_to_jd(t_obs)) - mjd)*3600.0*24.0 + 2.0*3600.0 #maybe not gmt
    t_sort = np.argsort(t_obs)
    t_obs = t_obs[t_sort]
    r_obs = r_obs[t_sort]
    v_obs = v_obs[t_sort]

    t_select = t_obs < SIM_TIME*3600.0

    t_obs = t_obs[t_select]
    r_obs = r_obs[t_select]
    v_obs = v_obs[t_select]

    #create RV_T plots, 

    fig, axs = plt.subplots(2,1,figsize=(12,8),dpi=80)
    axs[0].hist(sim.population['m'])
    axs[0].set(xlabel='Mass $m$ [kg]', ylabel='Counts')
    axs[1].hist(sim.population['d'])

    axs[1].set(xlabel='Diameter $d$ [m]', ylabel='Counts')

    fig.savefig(sim_root + '/{}_dist_param.png'.format(branch_name),bbox_inches='tight')
    
    fig, axs = plt.subplots(2,1,figsize=(12,8),dpi=80)
    axs[0].hist(sim.population['a']*(1.0 - sim.population['e']))
    axs[0].set(xlabel='Periapsis $q$ [km]', ylabel='Counts')
    axs[1].hist(sim.population['a']*(1.0 + sim.population['e']))
    axs[1].set(xlabel='Apoapsis $Q$ [km]', ylabel='Counts')

    fig.savefig(sim_root + '/{}_dist_qQ.png'.format(branch_name),bbox_inches='tight')
    

    figo, axo = sim.population.plot_distribution('orbits')
    figo.savefig(sim_root + '/{}_orbit_dist.png'.format(branch_name),bbox_inches='tight')
    
    tracklets = sim.catalogue.tracklets

    station = sim.radar._tx[0].ecef

    figall, axall = plt.subplots(2,1,figsize=(14,10),dpi=80)

    figp, axp = plt.subplots(2,1,figsize=(14,10),dpi=80)

    dets = {'t': [], 'r': [], 'v': [], 'index': []}

    for tracklet in tracklets:

        s_obj = sim.population.get_object(tracklet['index'])
        states = s_obj.get_state(tracklet['t'])
        num = len(tracklet['t'])
        rel_pos = states[:3,:]
        rel_vel = states[3:,:]
        range_v = np.empty((num,), dtype=np.float64)
        velocity_v = np.empty((num,), dtype=np.float64)
        angle_v = np.empty((num,), dtype=np.float64)

        _, k0 = radar._tx[0].scan.antenna_pointing(0.0)

        for ind in range(num):
            rel_pos[:,ind] -= station
            angle_v[ind] = coord.angle_deg(rel_pos[:,ind], k0)
            range_v[ind] = np.linalg.norm(rel_pos[:,ind])
            velocity_v[ind] = -np.dot(rel_pos[:,ind],rel_vel[:,ind])/range_v[ind]

        min_ind = np.argmin(angle_v)

        style = 'xb'
        alpha = 0.3

        axall[0].plot(tracklet['t']/3600.0, range_v*1e-3,
            '-b',
            alpha=0.5,
        )
        axall[1].plot(tracklet['t']/3600.0, velocity_v*1e-3,
            '-b',
            alpha=0.5,
        )


        axp[0].plot(tracklet['t'][min_ind]/3600.0, range_v[min_ind]*1e-3,
            style,
            alpha=alpha,
        )
        axp[1].plot(tracklet['t'][min_ind]/3600.0, velocity_v[min_ind]*1e-3,
            style,
            alpha=alpha,
        )

        dets['t'].append( tracklet['t'][min_ind] )
        dets['r'].append( range_v[min_ind]*1e-3 )
        dets['v'].append( velocity_v[min_ind]*1e-3 )
        dets['index'].append( tracklet['index'] )

    #dets['t'] = np.array(dets['t'])
    #dets['r'] = np.array(dets['r'])
    #dets['v'] = np.array(dets['v'])
    #dets['index'] = np.array(dets['index'], dtype=np.int64)

    r_lims = [np.min(np.array(dets['r'])), np.max(np.array(dets['r']))]
    v_lims = [np.min(np.array(dets['v'])), np.max(np.array(dets['v']))]

    axp[0].plot(t_obs/3600.0, r_obs, '.r', alpha=0.1)
    axp[1].plot(t_obs/3600.0, v_obs, '.r', alpha=0.1)

    axall[0].plot(t_obs/3600.0, r_obs, '.r', alpha=0.1)
    axall[1].plot(t_obs/3600.0, v_obs, '.r', alpha=0.1)


    axp[0].set(
        ylabel='Range [km]',
        xlabel='Time past $t_0$ [h]',
        title='Simulated Microsat-R debris',
    )
    axp[1].set(
        ylabel='Range-rate [km/s]',
        xlabel='Time past $t_0$ [h]',
    )
    figp.savefig(sim_root + '/{}_rv_vs_t_points.png'.format(branch_name),bbox_inches='tight')

    axall[0].set(
        ylabel='Range [km]',
        xlabel='Time past $t_0$ [h]',
        title='Simulated Microsat-R debris',
    )
    axall[1].set(
        ylabel='Range-rate [km/s]',
        xlabel='Time past $t_0$ [h]',
    )
    figall.savefig(sim_root + '/{}_rv_vs_t.png'.format(branch_name),bbox_inches='tight')



    #do mapping examination

    am_ratio = pop['A']/pop['m']
    am_size = np.max(am_ratio) - np.min(am_ratio)
    am_min = np.min(am_ratio)
    dv = pop['dV']
    dv_size = np.max(dv) - np.min(dv)
    dv_min = np.min(dv)

    parts_am = 2
    parts_dv = 3

    fig, axs = plt.subplots(parts_am*2,parts_dv,figsize=(16,12),dpi=80, sharex=True, sharey=False)

    for ami in range(0, parts_am*2, 2):
        for dvi in range(parts_dv):

            am_low_lim = (am_min + float(ami)*am_size/float(parts_am))
            am_hig_lim = (am_min + float(ami+1)*am_size/float(parts_am))
            inds_am = np.logical_and(
                am_ratio >= am_low_lim,
                am_ratio <= am_hig_lim
            )
            
            dv_low_lim = (dv_min + float(dvi)*dv_size/float(parts_dv))
            dv_hig_lim = (dv_min + float(dvi+1)*dv_size/float(parts_dv))
            inds_dv = np.logical_and(
                dv >= dv_low_lim,
                dv <= dv_hig_lim
            )
            inds = np.logical_and(inds_am, inds_dv)
            _inds = np.arange(len(pop), dtype=np.int64)
            inds = _inds[inds].tolist()
            #axs[ami, dvi].plot(t_obs/3600.0, r_obs, '.r', alpha=0.1, label='Observations')
            #axs[ami+1, dvi].plot(t_obs/3600.0, v_obs, '.r', alpha=0.1, label='Observations')

            axs[ami, dvi].plot(
                [_x/3600.0 for _i, _x in enumerate(dets['t']) if dets['index'][_i] in inds],
                [_x for _i, _x in enumerate(dets['r']) if dets['index'][_i] in inds],
                'xb',
                label=_label,
                alpha=1.0,
            )
            axs[ami+1, dvi].plot(
                [_x/3600.0 for _i, _x in enumerate(dets['t']) if dets['index'][_i] in inds],
                [_x for _i, _x in enumerate(dets['v']) if dets['index'][_i] in inds],
                'xb',
                label=_label,
                alpha=1.0,
            )

            axs[ami, dvi].set(
                ylabel='Range [km]',
                xlabel='Time past $t_0$ [h]',
                title='$\Delta V \in$ [{:2.1e}, {:2.1e}] m/s'.format(
                        dv_low_lim, 
                        dv_hig_lim,
                    ),
            )
            axs[ami, dvi].set_xlim(0., SIM_TIME)
            axs[ami, dvi].set_ylim(r_lims[0], r_lims[1])

            axs[ami+1, dvi].set_xlim(0., SIM_TIME)
            axs[ami+1, dvi].set_ylim(v_lims[0], v_lims[1])

            axs[ami+1, dvi].set(
                ylabel='Range-rate [km/s]',
                xlabel='Time past $t_0$ [h]',
                title='$A/m \in$ [{:2.1e}, {:2.1e}] m$^2$/kg'.format(
                        am_low_lim,
                        am_hig_lim
                    ),

            )

    #plt.legend()
    fig.savefig(sim_root + '/{}_correspondance_sim.png'.format(branch_name),bbox_inches='tight')


