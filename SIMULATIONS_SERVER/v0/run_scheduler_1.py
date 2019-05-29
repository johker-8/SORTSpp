#!/usr/bin/env python
#
#
from sim_config_1 import sim
import scheduler_library as sch


sim.load_detections() #to just load
sim.scan_statistics(save=True)
sim.sim_status()
sim._scheduler = sch.isolated_static_sceduler
sim.schedule_tracking()
sim.detection_statistics(save=True,version='_stupid_scheduler')
