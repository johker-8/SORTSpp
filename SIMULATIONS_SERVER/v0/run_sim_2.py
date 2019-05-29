#!/usr/bin/env python
#
#
from sim_config_2 import sim

sim.sim_status()

sim.run_scan(t=24.0)

#sim.load_detections(par=True) #to print
#sim.load_detections() #to just load
#sim.sim_status()
#sim.schedule_tracking()
#sim.detection_statistics(save=True)
