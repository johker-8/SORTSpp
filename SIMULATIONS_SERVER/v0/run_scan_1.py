#!/usr/bin/env python
#
#
from sim_config_1 import sim
import scheduler_library as sch
#import pdb

part = 0


if part==0:
	sim.scheduler = sch.dynamic_sceduler
	dt = 12.0
	for ti in range(6):
		sim.run_scan(t=dt)
		sim.load_detections(par=True) #to print
		sim.sim_status(print_to='run_scan-step-%i'%(ti))

		sim.scheduler_data['t0'] = dt*ti
		sim.scheduler_data['t1'] = dt*(ti+1)
		sim.schedule_tracking()

		sim.maintain_discovered()

		sim.print_tracks()
		sim.detection_statistics(save=True,version='_dynamic_sceduler_step%i'%(ti))
		sim.maintenance_statistics(save=True,version='_dynamic_sceduler_step%i'%(ti))
	sim.maint_statistics(save=True)
	sim.scan_statistics(save=True)
	
if part==1:
	sim.load_schedule()
	sim.sim_status(print_to='print_stats')
	sim.print_maintenance()
	sim.print_detections()
	sim.print_tracks() 

