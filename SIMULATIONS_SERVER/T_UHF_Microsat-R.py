#!/usr/bin/env python
#
#
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
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

os.environ['TZ'] = 'GMT'
time.tzset()

####### RUN CONFIG #######
part = 2
source = 'sim' #'real' or 'sim'
campagin = 1
prop = 1 #0 = SGP4, 1 = Orekit
##########################



if prop==0:
    prop = 'SGP4'
    _prop = propagator_sgp4.PropagatorSGP4
    _opts = {
        'out_frame': 'ITRF',
        'polar_motion': False,
    }
elif prop==1:
    prop = 'Orekit'
    _prop = propagator_orekit.PropagatorOrekit
    _opts = {
        'in_frame': 'TEME',
        'out_frame': 'ITRF',
        'drag_force': False,
    }

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
        el = 75.0,
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

branch_name = source + the_date + prop

sim_root = '/ZFS_DATA/SORTSpp/FP_sims/T_UHF_MicrosatR'

#initialize the radar setup 
radar = rlib.eiscat_uhf()

radar.set_FOV(max_on_axis=25.0, horizon_elevation=50.0)
radar.set_SNR_limits(min_total_SNRdb=14.0, min_pair_SNRdb=14.0)


radar.set_scan(scan)

fname = sim_root + '/{}_population.h5'.format(branch_name)
if os.path.exists(fname):
    pop = Population.load(fname,
        propagator = _prop,
        propagator_options = _opts,
    )
else:
    if source == 'real':
        if comm.rank == 0:
            pop = plib.Microsat_R_debris(
                mjd = mjd,
                num=200,
                radii_range=[0.025, 0.5],
                mass_range=[0.05, 10.0],
                propagator = _prop,
                propagator_options = _opts,
            )
            pop.save(fname)
        comm.barrier()
        if comm.rank != 0:
            pop = Population.load(fname,
                propagator = _prop,
                propagator_options = _opts,
            )
    elif source == 'sim':
        pop = plib.simulate_Microsat_R_debris(
            num=2000,
            max_dv=1e3,
            C_D_range = [1.8, 2.8],
            radii_range=[0.005, 0.2],
            mass_range=[0.01, 5.0],
            seed=27364872,
            propagator = _prop,
            propagator_options = _opts,
            mjd = mjd,
        )
        if comm.rank == 0:
            pop.save(fname)
    

data_file = './data/microsatR_uhf.h5'

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
