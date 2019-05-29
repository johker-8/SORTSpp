#!/usr/bin/env python
#
#
from sim_config_1_s4 import sim
import scheduler_library as sch
import matplotlib.pyplot as plt
#import pdb

sim.scheduler = sch.dynamic_sceduler
part = 3
ti = 1
dt = 6.0

if part==1:
    sim.clear_simulation()
    sim.run_scan(t=dt)
    sim.load_detections(par=True) #to save
    sim.sim_status(print_to='run_scan-step-%i'%(ti))
    sim.maint_statistics(save=True)
    sim.scan_statistics(save=True)

if part==1.5:
    sim.load_detections()
    sim.sim_status()
    sim.print_detections()

if part==2:
    sim.load_detections()
    sim.sim_status()
    sim.scheduler_data['t0'] = dt*(ti-1)
    sim.scheduler_data['t1'] = dt*ti
    #sim.scheduler_data['que_function'] = sch.que_value_dyn_v1
    #sim.scheduler_data['dt_sigma'] = 60*3 #pm 3 minutes
    #sim.scheduler_data['N_sigma'] = 30

    sim.scheduler_data['que_function'] = sch.que_value_dyn_v2
    
    sim.scheduler_data['N_rate'] = 20.
    sim.scheduler_data['dt_sigma'] = 2.*60.
    sim.scheduler_data['dt_offset'] = 0.*60.
    sim.scheduler_data['dt_sqew'] = 2.0
    sim.scheduler_data['tracklets_scale'] = 15.
    sim.scheduler_data['peak_snr_rate'] = 50.

    sim.scheduler_data['N_scale'] = 1.
    sim.scheduler_data['dt_scale'] = 3.
    sim.scheduler_data['tracklets_rate'] = 2.
    sim.scheduler_data['peak_snr_scale'] = 0.5
    sim.scheduler_data['tracklet_completion_rate'] = 20.0

    sim.schedule_tracking()

    load = n.array(sim._catalogue._system_load)
    plt.plot(load[:,0],load[:,1])
    plt.show()
    
    sim.print_tracks()

if part==3:
    sim.load_schedule()
    sim.print_tracks()
    sim.track_statistics(save=True,version='_dynamic_sceduler_ti%i'%(ti,))
