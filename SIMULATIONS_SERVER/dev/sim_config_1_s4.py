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

sim_root = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/sim_dev_sched_2scan'

#initialize the radar setup
e3d = rl.eiscat_3d()

e3d.set_FOV(max_on_axis=25.0,horizon_elevation=30.0)
e3d.set_SNR_limits(min_total_SNRdb=2.0,min_pair_SNRdb=1.0)
e3d.set_TX_bandwith(bw = 1.0e6)
e3d.set_beam('TX', alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.2,a0=40.0,az1=0,el1=90.0) )
e3d.set_beam('RX', alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.5,a0=40.0,az1=0,el1=90.0) )

#initialize the observing mode
e3d_scan = rslib.ns_fence_rng_model(min_el = 30.0, angle_step = 2.0, dwell_time = 0.1)

#3 by 3 grid at 300km 
az_points = n.arange(0,360,45).tolist() + [0.0];
el_points = [90.0-n.arctan(50.0/300.0)*180.0/n.pi, 90.0-n.arctan(n.sqrt(2)*50.0/300.0)*180.0/n.pi]*4+[90.0];
e3d_ionosphere = rslib.n_const_pointing_model(az_points,el_points,len(az_points),dwell_time = 0.4)

e3d_scan.set_radar_location(e3d)
e3d.set_scan(SST=e3d_scan,secondary_list=[e3d_ionosphere])

#load the input population
pop = p.master_catalog_factor(treshhold=1e-2,seed=12345)
pop.filter('i',lambda x: x >= 45.0)
pop._objs = pop._objs[:50,:]

sim = s.simulation( \
	radar = e3d,\
	population = pop,\
	sim_root = sim_root,\
	simulation_name = s.auto_sim_name('FIN_ns_rng_fence_masterf_sst1')
	)

# 25% duty, we get 10% for tracking, use all of it for tracking none for our own scan
#but we get data accsess to the interleaved experiment i.e. piggyback on ionospheric scan
sim.calc_observation_params(duty_cycle=0.25, \
    SST_fraction=0.1, \
    tracking_fraction=1.0, \
    interleaving_time_slice = e3d_ionosphere.dwell_time(), \
    scan_during_interleaved = True)
#coher_int_t=0.2

#sim._catalogue._known[:50] = True

sim._max_dpos = 50.0
sim._verbose = True
