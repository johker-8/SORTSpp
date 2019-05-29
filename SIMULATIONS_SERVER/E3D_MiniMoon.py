#!/usr/bin/env python
#
#
from mpi4py import MPI
comm = MPI.COMM_WORLD

import sys
import os
import time
import pickle

sys.path.insert(0, "/home/danielk/IRF/IRF_GITLAB/SORTSpp")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#SORTS Libraries
import radar_library as rlib
import population_library as plib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib

from population import Population
from propagator_rebound import PropagatorRebound

import simulate_tracking
import plothelp
import dpt_tools as dpt


####### RUN CONFIG #######
SIM_TIME = 3600.0*24.0*100.0
##########################

sim_root = '/home/danielk/IRF/E3D_PA/FP_sims/MiniMoon'

#initialize the radar setup 
radar = rlib.eiscat_3d(beam='interp', stage=1)

radar.set_FOV(max_on_axis=90.0, horizon_elevation=30.0)
radar.set_SNR_limits(min_total_SNRdb=1e-3, min_pair_SNRdb=1e-3)
radar.set_TX_bandwith(bw = 1.0e6)

#h_magnitude -> diam
# albedo = 0.14
# kneos formula 

pop = plib.NESCv9_mini_moons(
    num = 1,
    albedo = 0.14,
    synchronize=False,
    propagate_to = 384400e3,
    truncate = slice(100, None),
)

if comm.rank == 0:
    pop.save(sim_root + '/population.h5')

my_inds = list(range(comm.rank, len(pop), comm.size))

gen = pop.object_generator()
passes = [None]*len(pop)

t0 = time.time()
cnt = 0

for ind, sobj in enumerate(gen):
    if ind in my_inds:
        cnt += 1
        t_elaps = time.time() - t0
        print('Object {} of {}: {} h elapsed, estimated time left {} h'.format(
            ind,
            len(pop),
            t_elaps/3600.0,
            t_elaps/float(cnt)*(len(my_inds) - cnt)/3600.0,
        ))
        pass_struct = simulate_tracking.get_passes(sobj, radar, 0.0, SIM_TIME, t_samp=60.0)
        passes[ind] = pass_struct

if comm.rank == 0:
    for thr_id in range(1,comm.size):
        for ind in range(thr_id, len(pop), comm.size):
            passes[ind] = comm.recv(source=thr_id, tag=ind)
else:
    for ind in my_inds:
        comm.send(passes[ind], dest=0, tag=ind)

if comm.rank == 0:
    with open(sim_root + '/passes.pickle','wb') as pickle_out:
        pickle.dump(passes, pickle_out)

# analysis here
'''

with open(sim_root + '/passes.pickle','rb') as pickle_in:
    passes = pickle.load(pickle_in)
'''