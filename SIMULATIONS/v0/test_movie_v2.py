#!/usr/bin/env python
#
#
from sim_movie_v2 import sim
import scheduler_library as sch

sim.scheduler = sch.dynamic_sceduler
part = 3
ti = 1
dt = 1.0

if part==1:
    sim.run_scan(t=dt)
    sim.sim_status(print_to='run_scan-step-%i'%(ti))

if part==2:
    sim.load_detections()
    sim.scheduler_data['t0'] = dt*(ti-1)
    sim.scheduler_data['t1'] = dt*ti
    sim.scheduler_data['que_function'] = sch.que_value_dyn_v1

    sim.scheduler_data['dt_sigma'] = 1.*60.

    sim.schedule_tracking()

if part==3:
    sim.load_schedule()
    sim.schedule_movie(60.0/3600.0, dt = 0.5)
