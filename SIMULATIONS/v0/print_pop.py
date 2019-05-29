#!/usr/bin/env python
#
#
import numpy as n
from mpi4py import MPI
import sys

sys.path.insert(0, "/home/danielk/IRF/IRF_GITLAB/SORTSpp")

import population as p

#SORTS Libraries
import radar_library as rl
import radar_scan_library as rslib
import scheduler_library as sch
import antenna_library as alib

pop = p.master_catalog_factor(treshhold=1e-2,seed=12345)
pop.filter('i',lambda x: x >= 45.0)

print(len(pop))