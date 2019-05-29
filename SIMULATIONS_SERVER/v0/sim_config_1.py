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

#sim_root = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/sim_ew_fence_master'
sim_root = '/ZFS_DATA/SORTSpp/sim_ew_fence_master'

#initialize the radar setup
e3d = rl.eiscat_3d()
e3d._max_on_axis=25.0
e3d._min_SNRdb=5.0

#initialize the observing mode
e3d_scan = rslib.ew_fence_model(min_el = 30, angle_step = 1, dwell_time = 0.1)


e3d_scan.set_radar_location(e3d)
e3d._tx[0].scan = e3d_scan

#rc.plot_radar_conf(e3d)

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
	simulation_name = s.auto_sim_name('EW_FENCE')
	)

sim._verbose = False
