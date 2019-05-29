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

import propagator_sgp4
import propagator_orekit
import dpt_tools as dpt

####### RUN CONFIG #######
part = 3 #1 run obs, 2 run scheduler, 3 plots
SIM_TIME = 24.0
tx_power = '500kW' #'100kW' or '500kW'
prop = 'SGP4' #'SGP4' or 'Orekit'
freq = '1.2GHz' #'1.2GHz' or '2.4GHz'
##########################

sim_root = '/ZFS_DATA/SORTSpp/FP_sims/TSR_beampark'

if freq == '1.2GHz':
    _freq = 1.2e9
elif freq == '2.4GHz':
    _freq = 2.4e9

#initialize the radar setup 
radar = rlib.tromso_space_radar(freq = _freq)

if tx_power == '100kW':
    radar._tx[0].tx_power = 100e3
elif tx_power == '500kW':
    radar._tx[0].tx_power = 500e3

if prop == 'SGP4':
    _prop = propagator_sgp4.PropagatorSGP4
    _opts = {
        'out_frame': 'ITRF',
        'polar_motion': True,
    }
elif prop == 'Orekit':
    _prop = propagator_orekit.PropagatorOrekit
    _opts = {
        'in_frame': 'TEME',
        'out_frame': 'ITRF',
    }

branch_name = 'MASTER2009_' + prop + '_' + tx_power + '_' + freq

#load the input population
pop = plib.filtered_master_catalog_factor(
    radar = radar,
    treshhold = 0.001,
    min_inc = 50,
    prop_time = 48.0,
    propagator = _prop,
    propagator_options = _opts,
    detectability_file = sim_root + '/' + branch_name + '.h5',
)

sim = Simulation(
    radar = radar,
    population = pop,
    root = sim_root,
    scheduler = schlib.dynamic_scheduler,
    simulation_name = 'TSR beampark of MASTER2009'
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
    pass_dt=0.05,
)


################## RUNNING #####################

if part == 1:
    sim.set_version(branch_name)
    sim.run_observation(SIM_TIME)

    sim.status(fout='{}h_to_{}h'.format(0, int(SIM_TIME)))

    sim.print_maintenance()
    sim.print_detections()


if part == 2:
    sim.set_version(branch_name)
    sim.load()
    
    sim.set_scheduler_args(
        logger = sim.logger,
    )

    sim.run_scheduler()
    sim.print_tracks()
    sim.print_tracklets()

    sim.status(fout='{}h_to_{}h_scheduler'.format(0, int(SIM_TIME)))

if part == 3:
    sim.set_version(branch_name)
    sim.load()
    
    sim.plots()
    #sim.catalogue.track_statistics()
    #sim.catalogue.track_statistics_plot()
