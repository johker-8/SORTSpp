#!/usr/bin/env python
#
#
import numpy as n
from mpi4py import MPI
import sys

sys.path.insert(0, "/home/danielk/IRF/IRF_GITLAB/SORTSpp")
# SORTS imports CORE
import population as p
import simulation as s

#SORTS Libraries
import radar_library as rl
import radar_scan_library as rslib
import scheduler_library as sch
import antenna_library as alib

sim_root = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/mov_test_v2'

#initialize the radar setup
e3d = rl.eiscat_3d()

e3d.set_FOV(max_on_axis=25.0,horizon_elevation=30.0)
e3d.set_SNR_limits(min_total_SNRdb=10.0,min_pair_SNRdb=0.0)
e3d.set_TX_bandwith(bw = 1.0e6)
e3d.set_beam('TX', alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.2,a0=40.0,az1=0,el1=90.0) )
e3d.set_beam('RX', alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.5,a0=40.0,az1=0,el1=90.0) )

#initialize the observing mode
e3d_scan = rslib.ns_fence_rng_model(min_el = 30.0, angle_step = 2.0, dwell_time = 0.1)

e3d_scan.set_radar_location(e3d)
e3d.set_scan(e3d_scan)

#load the input population
pop = p.filtered_master_catalog_factor(e3d,treshhold=1e-2,seed=12345,filter_name='e3d_planar_beam')
pop._objs = pop._objs[:2000,:]

sim = s.simulation( \
    radar = e3d,\
    population = pop,\
    sim_root = sim_root,\
    simulation_name = s.auto_sim_name('mov_test_v2')
    )

sim.calc_observation_params(duty_cycle=0.25, SST_fraction=1.0, tracking_fraction=0.5, interleaving_time_slice = 0.4, SST_time_slice=0.2)

sim._max_dpos = 50.0
sim._verbose = True

for i in range(1000):
	sim._catalogue._known[i] = True
