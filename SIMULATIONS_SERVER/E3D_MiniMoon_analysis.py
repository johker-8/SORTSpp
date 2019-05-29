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
SIM_TIME = 3600.0*24.0*30.0
##########################

#how many
#hist over diameter
#hist over detection rate / day
#hist over detection times after "closeness"

#first detection time -> propagate from that time -> see how long they stay captured after
#what is captured? e < 1 in earth centric coordinates

sim_root = '/home/danielk/IRF/E3D_PA/FP_sims/MiniMoon'

txi = 0
pop = Population.load(sim_root + '/population.h5')

radar = rlib.eiscat_3d(beam='interp', stage=1)

with open(sim_root + '/passes.pickle','rb') as pickle_in:
    passes = pickle.load(pickle_in)

print(passes)

det_n = np.sum([len(p['snr'][txi]) for p in passes if len(p['snr']) > 0])

do_print = True

det_snr = np.empty((det_n, 3), dtype=np.float64)

for ind, pass_data in enumerate(passes):
    if len(pass_data["snr"]) > 0:
        for pi, pas in enumerate(pass_data["snr"][txi]):
            if do_print:
                print('Detection {}:'.format(pi))
            for rxi, rx in enumerate(radar._rx):
                det_snr[pi, rxi] = passes["snr"][txi][pi][rxi][0]
                if do_print:
                    print('|- {:<14}: {:<7.4f} SNRdB'.format(rx.name, 10.0*np.log10(det_snr[pi, rxi])))

print('TOTAL: {} / {} detected'.format(det_n, len(passes)))

fig, ax = plt.subplots(1,1,figsize=(12,8),dpi=80)
ax.hist(np.max(det_snr, axis=1))
ax.set(
    xlabel='SNRdB',
    ylabel='Frequency',
    title='{} detections of {}'.format(radar.name, pop.name),
)


plt.show()