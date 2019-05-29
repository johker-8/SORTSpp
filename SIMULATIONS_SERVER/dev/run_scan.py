#!/usr/bin/env python
#
#
from sim_config_dev import sim
import scheduler_library as sch
#import pdb

part = 1

if part==0:
	sim.clear_simulation()

if part==1:
	sim.run_scan(t=24.0)
	sim.load_detections(par=True) #to print
	sim.sim_status(print_to='run_scan-v0')

if part==2:
	sim.load_detections() #to print
	sim.scan_statistics(save=True)
	sim._scheduler = sch.memory_static_sceduler
	sim._scheduler_data['memory_track_time'] = 1e-3 #1 minute -> good orbit?
	sim.schedule_tracking()
	sim.detection_statistics(save=True,version='_memory_scheduler')

if part==-1:
	sim.load_schedule()
	sim.sim_status()
	sim.print_detections()
	sim.print_tracks()

if part==3:
	sim.load_schedule()
	sim.sim_status(print_to='prior_tests-v0')
	sim.generate_tracklets()
	sim.save_tracklets()

if part==4:
	#pdb.set_trace()
	sim.load_schedule()
	sim.sim_status(print_to='prior_tests-v1')
	sim.discover_orbits()
	sim.save_tracklets()

if part==5:
	sim.load_schedule()
	sim.set_SMART_location('/home/danielk/IRF/IRF_GITLAB/SORTSpp/smart/wrk_sa')
	sim.run_orbit_determination()

if part==6:
	sim.calculate_orbit_errors()