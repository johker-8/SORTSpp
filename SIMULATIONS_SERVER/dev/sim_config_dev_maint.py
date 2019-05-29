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

#sim_root = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/sim_dev_maint_v2'
sim_root = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/sim_dev_maint'

#initialize the radar setup
e3d = rl.eiscat_3d()

e3d.set_FOV(max_on_axis=25.0,horizon_elevation=30.0)
e3d.set_SNR_limits(min_total_SNRdb=2.0,min_pair_SNRdb=1.0)
e3d.set_TX_bandwith(bw = 1.0e6)
e3d.set_beam('TX', alib.e3d_array_beam_stage1() )
e3d.set_beam('RX', alib.e3d_array_beam() )

#initialize the observing mode
e3d_scan = rslib.ew_fence_model(min_el = 30, angle_step = 1, dwell_time = 0.1)

e3d_scan.set_radar_location(e3d)
e3d.set_scan(e3d_scan)

#load the input population
pop = p.master_catalog_factor(treshhold=1e-2,seed=12345)
pop.filter('i',lambda x: x >= 45.0)


pros_n = 7
n_obj = pros_n*2
#n_obj = pros_n*2
pop._objs = pop._objs[:n_obj,:]

sim = s.simulation( \
	radar = e3d,\
	population = pop,\
	sim_root = sim_root,\
	simulation_name = s.auto_sim_name('EW_FENCE_DEV_MAINT')
	)

sim.calc_observation_params(duty_cycle=0.25, SST_fraction=1.0, tracking_fraction=0.5, interleaving_time_slice = 0.4)


#set NORAD catalouge as known and in maintinence mode
#space debries is in scaning mode
#for I,ID in enumerate(pop._objs[:,0]):
#	if int(n.floor(ID/1e5)) == 2101:
#		sim._catalogue._known[I] = True

#for I in range(pros_n*1):
#	sim._catalogue._known[I] = True
sim._max_dpos = 50.0

sim._verbose = True
