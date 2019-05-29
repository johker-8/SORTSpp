#!/usr/bin/env python
#
#

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
import population_library as plib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib
import plothelp
import dpt_tools as dpt
import ccsds_write

SIM_TIME = 24.0*4

sim_root = '/home/danielk/test_SORTSpp_sim'

pop = plib.master_catalog()
pop.delete(slice(25, None))

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
    simulation_name = 'Example simulation',
)

#sim.set_log_level(logging.DEBUG)

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

if part == 2:
    sim.load()
    sim.generate_tracklets()

if part == 3:
    sim.load()
    sim.generate_priors(
        frame_transformation='TEME',
        tracklet_truncate = slice(None, None, 20),
    )

if part == 4:
    sim.load()
    sim.run_orbit_determination(
        frame_transformation = 'TEME',
        error_samp = 500,
        steps = 10000, 
        max_zenith_error = 0.9,
        tracklet_truncate = slice(None, None, 400),
        tune = 1000,
    )
