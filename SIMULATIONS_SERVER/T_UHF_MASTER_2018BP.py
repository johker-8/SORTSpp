#!/usr/bin/env python
#
#
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
import os
import time

# replace this with the path to your sorts
#sys.path.insert(0, "/home/danielk/PYTHON/SORTSpp")
sys.path.insert(0, "/home/danielk/IRF/IRF_GITLAB/SORTSpp")

# SORTS imports CORE
import population_library as plib
from simulation import Simulation

#SORTS Libraries
import radar_library as rlib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib

import numpy as np
import dpt_tools as dpt

os.environ['TZ'] = 'GMT'
time.tzset()

dt = np.datetime64('2019-04-02T12:01')
mjd = dpt.npdt2mjd(dt)

####### RUN CONFIG #######
part = 3
SIM_TIME = 24.0
##########################

#sim_root = '/ZFS_DATA/SORTSpp/FP_sims/T_UHF_MicrosatR'
sim_root = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/T_UHF_2018BP'

#initialize the radar setup 
radar = rlib.eiscat_uhf()

radar.set_FOV(max_on_axis=25.0, horizon_elevation=50.0)
radar.set_SNR_limits(min_total_SNRdb=14.0, min_pair_SNRdb=14.0)

scan = rslib.beampark_model(
    lat = radar._tx[0].lat,
    lon = radar._tx[0].lon,
    alt = radar._tx[0].alt, 
    az = 90.0,
    el = 75.0,
)
radar.set_scan(scan)

#load the input population
pop = plib.filtered_master_catalog_factor(
    radar = radar,
    treshhold = 0.01,
    min_inc = 50,
    prop_time = 48.0,
)

sim = Simulation(
    radar = radar,
    population = pop,
    root = sim_root,
    scheduler = schlib.dynamic_scheduler,
    simulation_name = 'EISCAT UHF 2018 Beampark'
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
    sim.run_observation(SIM_TIME)

    sim.status(fout='{}h_to_{}h'.format(0, int(SIM_TIME)))
    sim.print_detections()


if part == 2:
    sim.set_version('scheduler')
    sim.checkout_simulation('master') #load included

    sim.set_scheduler_args(
        logger = sim.logger,
    )

    sim.run_scheduler()
    sim.print_tracks()
    sim.print_tracklets()

    sim.status(fout='{}h_to_{}h_scheduler'.format(0, int(SIM_TIME)))
    
    sim.plots()

if part == 3:
    sim.set_version('scheduler')
    sim.load()
    
    tracklets = sim.catalogue.tracklets

    station = sim.radar._tx[0].ecef

    fig, ax = plt.subplots(2,1,figsize=(12,8),dpi=80)

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

        if tracklet['index'] == 0:
            style = 'or'
            alpha = 1
        else:
            style = '.b'
            alpha = 0.01

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
    fig.savefig(sim._plots_folder + '/sim_rv_vs_t.png',bbox_inches='tight')
    plt.show()

if part == 4:
    sim.set_version('scheduler')
    sim.load()
    
    sim.generate_tracklets()
    sim.generate_priors()