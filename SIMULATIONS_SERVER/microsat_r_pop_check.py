#!/usr/bin/env python
#
#
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import sys
import os
import time

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

sim_root = '/ZFS_DATA/SORTSpp/FP_sims/T_UHF_MicrosatR'

bname = 'real' + '_2019_04_02_' + 'SGP4'

fname = sim_root + '/{}_population.h5'.format(bname)

pop = Population.load(fname)

print(pop)
