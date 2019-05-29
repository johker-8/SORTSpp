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

### 
# PART
##
part = 2


#Project 5: Lets simulate our system!

from DEV_demo1 import SC
from DEV_demo2 import radar
radar.set_scan(SC)

pop = plib.master_catalog()

pop.delete(slice(100, None)) 
#46 was detecable, but lets use the 100 since we didnt save the results

SIM_TIME = 24.0*1

sim_root = '/home/danielk/IRF/E3D_PA/DEMO'

sim = Simulation(
    radar = radar,
    population = pop,
    root = sim_root,
    scheduler = schlib.dynamic_scheduler,
    simulation_name = 'Demo simulation',
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
