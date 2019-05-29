#!/usr/bin/env python
#
#
from sim_config_5 import sim
import scheduler_library as sch

sim.load_detections() #to print
sim.sim_status()
sim._scheduler = sch.memory_static_sceduler
sim._scheduler_data['memory_track_time'] = 60.0 #3 pulses of 10ms each maybe,1 minute -> good orbit?
sim.schedule_tracking()
sim.detection_statistics(save=True,version='_memory_scheduler_60s')

sim._scheduler_data['memory_track_time'] = 10.0
sim.schedule_tracking()
sim.detection_statistics(save=True,version='_memory_scheduler_10s')

sim._scheduler_data['memory_track_time'] = 1.0
sim.schedule_tracking()
sim.detection_statistics(save=True,version='_memory_scheduler_1s')

sim._scheduler_data['memory_track_time'] = 30e-3
sim.schedule_tracking()
sim.detection_statistics(save=True,version='_memory_scheduler_30e-3s')

