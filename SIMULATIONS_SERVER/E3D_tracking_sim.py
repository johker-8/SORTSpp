#!/usr/bin/env python
#
#
import numpy as np
from mpi4py import MPI
import sys

# replace this with the path to your sorts
sys.path.insert(0, "/home/danielk/PYTHON/SORTSpp")

# SORTS imports CORE
import population_library as plib
from simulation import Simulation

#SORTS Libraries
import radar_library as rlib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib


####### RUN CONFIG #######
part = 3
SIM_TIME = 24.0*7.0
##########################

sim_root = '/ZFS_DATA/SORTSpp/FP_sims/E3D_tracking'

#initialize the radar setup
radar = rlib.eiscat_3d(beam='interp', stage=1)

radar.set_FOV(max_on_axis=90.0, horizon_elevation=30.0)
radar.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=1.0)
radar.set_TX_bandwith(bw = 1.0e6)

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
    simulation_name = 'E3D tracking of MASTER2009'
)

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

if part == 1:
    sim.run_observation(SIM_TIME)

    sim.status(fout='{}h_to_{}h'.format(0, int(SIM_TIME)))

    sim.print_maintenance()
    sim.print_detections()


if part == 2:
    sim.set_version('scheduler_2d')
    sim.checkout_simulation('master') #load included

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


if part == 3:
    sim.set_version('scheduler_2d')
    sim.load()
    sim.plots()


if part == 4:
    sim.set_version('scheduler_2d')
    sim.load()

    sim.generate_tracklets()
    sim.generate_priors()
