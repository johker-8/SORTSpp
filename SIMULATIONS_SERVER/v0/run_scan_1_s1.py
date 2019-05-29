#!/usr/bin/env python
#
#
from sim_config_1_s1 import sim
import scheduler_library as sch
#import pdb

sim.scheduler = sch.dynamic_sceduler
part = 3
ti = 1
dt = 6.0

if part==1:
	sim.run_scan(t=dt)
	sim.load_detections(par=True) #to save
	sim.sim_status(print_to='run_scan-step-%i'%(ti))
	sim.maint_statistics(save=True)
	sim.scan_statistics(save=True)

if part==2:
	sim.load_detections()
	sim.sim_status()
	sim.maint_statistics(save=True)
	sim.scan_statistics(save=True)

if part==3:
	sim.load_detections()
	sim.sim_status()
	sim.scheduler_data['t0'] = dt*(ti-1)
	sim.scheduler_data['t1'] = dt*ti
	sim.schedule_tracking()

	sim.maintain_discovered()

	sim.print_tracks()
	sim.detection_statistics(save=True,version='_dynamic_sceduler_step%i'%(ti))
	sim.maintenance_statistics(save=True,version='_dynamic_sceduler_step%i'%(ti))
