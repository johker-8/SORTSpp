#!/usr/bin/env python
#
#
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys

# replace this with the path to your sorts
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

### SIM RUN CONFIG ###
part = 2.2
sim_time = 24.0
######################


sim_root = '/home/danielk/IRF/E3D_PA/tmp/MY_TEST'

#initialize the radar setup
radar = rlib.eiscat_3d()

radar.set_FOV(max_on_axis=25.0,horizon_elevation=30.0)
radar.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=1.0)


#load the input population
pop = plib.filtered_master_catalog_factor(
    radar = radar,
    treshhold = 0.01,
    min_inc = 50,
    prop_time=48.0,
)
pop.delete(slice(40,None))

sim = Simulation(
    radar = radar,
    population = pop,
    root = sim_root,
    scheduler = schlib.dynamic_scheduler,
)

if part == 0:
    sim.clear_simulation()

if part == 1:
    if sim.check_load():
        sim.load()
        sim.branch_simulation('scheduler_run')
    else:
        sim.observation_parameters(
            tracking_fraction=0.25,
        )
        sim.catalogue.maintain(slice(10,None))

        sim.run_observation(sim_time)

        sim.status(fout='{}h_to_{}h'.format(0, int(sim_time)))

        sim.print_maintenance()
        sim.print_detections()


if part == 2:
    sim.set_version('scheduler_run')
    sim.checkout_simulation('master') #load included

    sim.set_scheduler_args(
        reward_function = rflib.rewardf_exp_peak_SNR,
        reward_function_config = {
            'sigma_t': 60.0*5.0,
        },
        logger = sim.logger,
    )

    sim.run_scheduler()
    sim.print_tracks()
    sim.print_tracklets()

if part == 2.1:
    sim.set_version('scheduler_run_different_args')
    sim.checkout_simulation('master') #load included

    sim.set_scheduler_args(
        reward_function = rflib.rewardf_exp_peak_SNR,
        reward_function_config = {
            'sigma_t': 60.0,
        },
        logger = sim.logger,
    )
    sim.run_scheduler()
    sim.print_tracks()
    sim.print_tracklets()

if part == 3.1:
    sim.set_version('new_radar')
    sim.radar._tx[0].tx_power = 500e3
    sim.run_observation(sim_time)


if part == 2.2:
    sim.set_version('scheduler_run')
    sim.load()

    plt.hist(sim.catalogue.tracks['t0'])
    plt.show()




if part == 3:
    sim.set_version('scheduler_run')
    sim.load()
    sim.print_tracks()
    sim.print_tracklets()
    sim.status(fout='{}h_to_{}h'.format(0, int(sim_time)))

if part == 4:
    sim.set_version('scheduler_run')
    sim.load()
    
    sim.plots()
    
#interactive python test
if part == 5:
    plt.ion()

    sim.set_version('scheduler_run')
    sim.load()

if part == 6:
    sim.set_version('scheduler_run')
    sim.load()
    
    sim.generate_tracklets()

if part == 7:
    sim.set_version('new_scheduler')
    sim.checkout_simulation('master')

    sim.set_scheduler_args(
        reward_function = rflib.rewardf_exp_peak_SNR,
        reward_function_config = {
            'sigma_t': 60.0*5.0,
        },
        logger = sim.logger,
    )

    sim.run_scheduler()

if part == 8:
    sim.set_version('scheduler_run')
    sim.load()
    
    sim.generate_priors()