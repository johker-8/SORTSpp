#!/usr/bin/env python
#
#
from sim_config_1 import sim
import scheduler_library as sch

sim.load_detections() #to print
sim.sim_status()
sim._scheduler = sch.memory_static_sceduler
sim._scheduler_data['memory_track_time'] = 0.1
sim.schedule_tracking()
sim.detection_statistics(save=True,version='_memory_scheduler_1e-1s')

