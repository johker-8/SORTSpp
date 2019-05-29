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
sim_root = '/ZFS_DATA/SORTSpp/sim_shared_ew_fence_master'

#initialize the radar setup
e3d = rl.eiscat_3d()
e3d._max_on_axis=25.0
e3d._min_SNRdb=5.0

#initialize the observing mode
#lets say you measure ionospheric parameters in 2 directions, integration time 0.4s
#and you want updates every second
#that gives 0.2s over for space debries every s
#if we use a dwell of 0.1 and angle step of 1 it is 2 deg every s
el_points_fence = rslib.calculate_fence_angles(min_el=30.0,angle_step=5.0)
az_points_fence = [90]*len(el_points_fence[el_points_fence > 90]) + [180]*len(el_points_fence[el_points_fence <= 90])
el_points_fence[el_points_fence > 90] = 180-el_points_fence[el_points_fence > 90]

dwells = []
az_points = []
el_points = []
for ind in range(len(el_points_fence)):
	dwells.append(0.4)
	az_points.append(0.0)
	el_points.append(-90.0)
	dwells.append(0.1)
	az_points.append(az_points_fence[ind])
	el_points.append(el_points_fence[ind])

e3d_scan = rslib.n_dyn_dwell_pointing_model(az_points,el_points,len(dwells),dwells)

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
	simulation_name = s.auto_sim_name('SHARED_EW_FENCE')
	)

sim._verbose = False