#!/usr/bin/env python
#
#
from sim_config_6 import sim

sim.sim_status()

sim.run_scan(t=24.0)

sim.load_detections(par=True) #to print
sim.sim_status()

