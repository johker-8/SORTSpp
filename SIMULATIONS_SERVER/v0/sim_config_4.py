#!/usr/bin/env python
#
#
import numpy as n
from mpi4py import MPI
import sys

#sys.path.insert(0, "/home/danielk/IRF/IRF_GITLAB/SORTSpp")
sys.path.insert(0, "/home/danielk/PYTHON/SORTSpp")
# SORTS imports
import population as p
import radar_library as rl
import radar_scan_library as rslib
import simulation as s
import scheduler_library as sch

#sim_root = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/sim_shared_ew_fence_master'
sim_root = '/ZFS_DATA/SORTSpp/sim_piggyback_ionospheric_scan'

#initialize the radar setup
e3d = rl.eiscat_3d()
e3d._max_on_axis=25.0
e3d._min_SNRdb=5.0

#initialize the observing mode
#lets say you measure ionospheric parameters in a 3-by-3 grid at 300km altitide separated by 50km directions, integration time 0.4s
#we piggyback a analysis on this, how good at discovery is it?
az_points = n.arange(0,360,45).tolist() + [0.0];
el_points = [90.0-n.arctan(50.0/300.0)*180.0/n.pi, 90.0-n.arctan(n.sqrt(2)*50.0/300.0)*180.0/n.pi]*4+[90.0];
e3d_scan = rslib.n_const_pointing_model(az_points,el_points,len(az_points),dwell_time = 0.4)

e3d_scan.set_radar_location(e3d)
e3d._tx[0].scan = e3d_scan

#load the input population
pop = p.master_catalog()
pop._objs = pop._objs[pop._objs[:,3] > 45.0,:]
pop._objs = pop._objs[pop._objs[:,8] > 1e-2,:]
pop._objs = pop._objs[pop._objs[:,2] < 1,:]

sim = s.simulation( \
	radar = e3d,\
	population = pop,\
	sim_root = sim_root,\
	scheduler = sch.isolated_static_sceduler,\
	simulation_name = s.auto_sim_name('PIGGYBACK_IONSPH_SCAN')
	)

sim._verbose = False