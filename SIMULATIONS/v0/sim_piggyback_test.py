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

sim_root = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/piggyback_test'

#initialize the radar setup
e3d = rl.eiscat_3d()

e3d.set_FOV(max_on_axis=25.0,horizon_elevation=30.0)
e3d.set_SNR_limits(min_total_SNRdb=10.0,min_pair_SNRdb=0.0)
e3d.set_TX_bandwith(bw = 1.0e6)
e3d.set_beam('TX', alib.e3d_array_beam_stage1(opt='dense') )
e3d.set_beam('RX', alib.e3d_array_beam() )

#initialize the observing mode
e3d_scan = rslib.ns_fence_rng_model(min_el = 30.0, angle_step = 2.0, dwell_time = 0.2)

#3 by 3 grid at 300km 
az_points = n.arange(0,360,45).tolist() + [0.0];
el_points = [90.0-n.arctan(50.0/300.0)*180.0/n.pi, 90.0-n.arctan(n.sqrt(2)*50.0/300.0)*180.0/n.pi]*4+[90.0];
e3d_ionosphere = rslib.n_const_pointing_model(az_points,el_points,len(az_points), dwell_time = 7.5)

e3d_scan.set_radar_location(e3d)
e3d.set_scan(SST=e3d_scan,secondary_list=[e3d_ionosphere])

#load the input population
pop = p.filtered_master_catalog_factor(e3d,treshhold=1e-2,seed=12345,filter_name='e3d_full_beam')
pop._objs = pop._objs[:2000,:]

sim = s.simulation( \
    radar = e3d,\
    population = pop,\
    sim_root = sim_root,\
    simulation_name = s.auto_sim_name('piggyback_test')
    )

sim.calc_observation_params(\
    duty_cycle=0.01, \
    SST_fraction=0.1, \
    tracking_fraction=1.0, \
    SST_time_slice=0.2, \
    interleaving_time_slice = 7.5, \
    scan_during_interleaved = True)

sim._max_dpos = 50.0
sim._verbose = True
