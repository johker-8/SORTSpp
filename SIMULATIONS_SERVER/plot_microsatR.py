#!/usr/bin/env python
#
#
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
import os
import time

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

os.environ['TZ'] = 'GMT'
time.tzset()

####### RUN CONFIG #######
plot_part = 0
##########################

prop = 'SGP4'
_prop = propagator_sgp4.PropagatorSGP4
_opts = {
        'out_frame': 'ITRF',
        'polar_motion': False,
}

branch_names = [
    'real' + '_2019_04_02_' + 'SGP4',
    'real' + '_2019_04_02_' + 'Orekit',
    'sim' + '_2019_04_02_' + 'SGP4',
    'sim' + '_2019_04_02_' + 'Orekit',
]

sim_root = '/ZFS_DATA/SORTSpp/FP_sims/T_UHF_MicrosatR'

#initialize the radar setup 
radar = rlib.eiscat_uhf()

radar.set_FOV(max_on_axis=25.0, horizon_elevation=50.0)
radar.set_SNR_limits(min_total_SNRdb=14.0, min_pair_SNRdb=14.0)

scan = rslib.beampark_model(
    lat = radar._tx[0].lat,
    lon = radar._tx[0].lon,
    alt = radar._tx[0].alt, 
    az = 90.0,
    el = 70.0,
)
radar.set_scan(scan)

sims = []

data_file = './data/microsatR_uhf.h5'

with h5py.File(data_file, 'r') as hf:
    t_obs = hf['t'].value
    r_obs = hf['r'].value
    v_obs = hf['v'].value

dt = np.datetime64('2019-04-02T12:01')
mjd = dpt.npdt2mjd(dt)

t_obs = (dpt.jd_to_mjd(dpt.unix_to_jd(t_obs)) - mjd)*3600.0*24.0 + 2.0*3600.0 #maybe not gmt
t_sort = np.argsort(t_obs)
t_obs = t_obs[t_sort]
r_obs = r_obs[t_sort]
v_obs = v_obs[t_sort]

t_select = t_obs < 24.0*3600.0

t_obs = t_obs[t_select]
r_obs = r_obs[t_select]
v_obs = v_obs[t_select]

for branch_name in branch_names:

    fname = sim_root + '/{}_population.h5'.format(branch_name)

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

    sim.set_version(branch_name)
    sim.load()
    
    sims.append(sim)


if plot_part == 0:

    for sim, branch_name in zip(sims, branch_names):
        fig, axs = plt.subplots(2,2,figsize=(12,8),dpi=80)
        axs[0,0].hist(sim.population['C_D'][1:])
        axs[0,0].set(xlabel='Drag Coeff $C_D$', ylabel='Frequency')
        axs[0,1].hist(sim.population['A'][1:])
        axs[0,1].set(xlabel='Area [m$^2$]', ylabel='Frequency')
        axs[1,0].hist(sim.population['m'][1:])
        axs[1,0].set(xlabel='Mass [kg]', ylabel='Frequency')
        axs[1,1].hist(sim.population['d'][1:])
        axs[1,1].set(xlabel='Diameter $d$ [m]', ylabel='Frequency')
        fig.savefig(sim_root + '/{}_dist_param.png'.format(branch_name),bbox_inches='tight')
        
        figo, axo = sim.population.plot_distribution('orbits')
        figo.savefig(sim_root + '/{}_orbit_dist.png'.format(branch_name),bbox_inches='tight')
        
        tracklets = sim.catalogue.tracklets

        station = sim.radar._tx[0].ecef

        fig, ax = plt.subplots(2,1,figsize=(12,8),dpi=80)

        figp, axp = plt.subplots(2,1,figsize=(12,8),dpi=80)

        for tracklet in tracklets:

            s_obj = sim.population.get_object(tracklet['index'])
            states = s_obj.get_state(tracklet['t'])
            num = len(tracklet['t'])
            rel_pos = states[:3,:]
            rel_vel = states[3:,:]
            range_v = np.empty((num,), dtype=np.float64)
            velocity_v = np.empty((num,), dtype=np.float64)
            for ind in range(num):
                rel_pos[:,ind] -= station

                range_v[ind] = np.linalg.norm(rel_pos[:,ind])
                velocity_v[ind] = np.dot(rel_pos[:,ind],rel_vel[:,ind])/range_v[ind]

            min_ind = np.argmin(range_v)

            style = '.b'
            alpha = 0.01

            axp[0].plot(tracklet['t'][min_ind]/3600.0, range_v[min_ind]*1e-3,
                style,
                alpha=alpha,
            )
            axp[1].plot(tracklet['t'][min_ind]/3600.0, velocity_v[min_ind]*1e-3,
                style,
                alpha=alpha,
            )

            ax[0].plot(tracklet['t']/3600.0, range_v*1e-3,
                style,
                alpha=alpha,
            )
            ax[1].plot(tracklet['t']/3600.0, velocity_v*1e-3,
                style,
                alpha=alpha,
            )

        ax[0].set(
            ylabel='Range [km]',
            xlabel='Time past $t_0$ [h]',
            title='Simulated Microsat-R debris',
        )
        ax[1].set(
            ylabel='Range-rate [km/s]',
            xlabel='Time past $t_0$ [h]',
        )
        fig.savefig(sim_root + '/{}_rv_vs_t.png'.format(branch_name),bbox_inches='tight')

        axp[0].plot(t_obs/3600.0, r_obs, '.r', alpha=0.2)
        axp[1].plot(t_obs/3600.0, v_obs, '.r', alpha=0.2)


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

