#!/usr/bin/env python
#
#
from sim_dynamic_coldstart_test import sim
import scheduler_library as sch

sim.scheduler = sch.dynamic_sceduler
part = 2
dt = 2.0
T_total = 6.0
num = T_total / dt


import logging
sim.set_log_level(logging.WARNING)

sim.scheduler_data['que_function'] = sch.que_value_dyn_v1
sim.scheduler_data['dt_sigma'] = 1.*60.

if part==1:

    for ti in range(1,int(num)+1):
        sim._logger.always('Running scan: {} h to {} h'.format(
            dt*(ti-1), dt*ti))
        sim.run_scan(t=dt)
        sim.sim_status(print_to='run_scan-step-%i'%(ti))
        sim.load_detections(par=True)
        sim.print_detections(mode='less')
        sim.print_maintenance(mode='less')

        sim.scheduler_data['t0'] = dt*(ti-1)
        sim.scheduler_data['t1'] = dt*ti
        sim.schedule_tracking()

if part==2:
    sim.load_schedule()
    sim.maint_statistics(save=True)
    sim.scan_statistics(save=True)

    t_curr = int(sim._sim_time())

    sim.track_statistics(save=True,version='_dynamic_sceduler_t%i'%(t_curr,))
    sim.detection_statistics(save=True,version='_dynamic_sceduler_t%i'%(t_curr,))
    sim.maintenance_statistics(save=True,version='_dynamic_sceduler_t%i'%(t_curr,))
    sim.print_tracks()
    sim.sim_status()

if part==3.1:
    sim.load_schedule()
    sim.generate_priors()
    sim.sim_status()

if part==3.2:
    sim.load_schedule()
    sim.generate_tracklets()
    sim.sim_status()

