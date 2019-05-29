#!/usr/bin/env python
#
#
from sim_config_dev_maint import sim
import scheduler_library as sch
#import pdb

part = -2

if part==123:
	sim.clear_simulation()
	sim.scheduler = sch.dynamic_sceduler

	for ti in range(2):
		sim.run_scan(t=5.0)
		sim.load_detections(par=True) #to print
		sim.sim_status(print_to='run_scan-step-%i-v2'%(ti))

		sim.scheduler_data['t0'] = 5.0*ti
		sim.scheduler_data['t1'] = 5.0*(ti+1)
		sim.schedule_tracking()

		sim.maintain_discovered()

		sim.print_tracks()
	#sim.detection_statistics(save=True,version='_dynamic_sceduler_v0')
	#sim.maintenance_statistics(save=True,version='_dynamic_sceduler_v0')

if part==-3:
	for I in range(len(sim._population)):
		sim._population.print_row(I)

if part==0:
	sim.clear_simulation()

if part==1:
	sim.run_scan(t=24.0)
	sim.load_detections(par=True) #to print
	sim.sim_status(print_to='run_scan-v0')

if part==1.2:
	sim.run_scan(t=10.0)
	sim.load_detections(par=True) #to print
	sim.sim_status(print_to='run_scan-step-1')

	for I in range(len(sim._population)):
		if sim._catalogue._detections[I] is not None:
			print("||||| ADDING OBJ %i TO MAINTAINED CATALOGUE |||||"%(I))
			sim._catalogue._known[I] = True

	sim.run_scan(t=14.0)
	sim.load_detections(par=True) #to print
	sim.sim_status(print_to='run_scan-step-2')


if part==-2:
	sim.load_schedule()
	sim.sim_status(print_to='run_scan-v0')
	sim.print_maintenance()
	sim.print_detections()
	sim.print_tracks() 
	sim.maint_statistics(save=True)
	sim.scan_statistics(save=True)

if part==2:
	sim.load_detections() #to print
	sim.scheduler = sch.dynamic_sceduler
	sim.schedule_tracking()
	sim.print_tracks()
	sim.detection_statistics(save=True,version='_dynamic_sceduler_v0')
	sim.maintenance_statistics(save=True,version='_dynamic_sceduler_v0')

if part==-1:
	sim.load_schedule()
	sim.sim_status()
	sim.print_detections()
	sim.print_maintinence()
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