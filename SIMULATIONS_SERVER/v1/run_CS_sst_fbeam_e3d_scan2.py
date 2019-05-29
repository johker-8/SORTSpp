#!/usr/bin/env python
#
#
from sim_CS_sst_fbeam_e3d_scan2 import sim
import scheduler_library as sch

sim.scheduler = sch.dynamic_sceduler
part = 1
tn = 4*7
dt = 6.0

sim.scheduler_data['que_function'] = sch.que_value_dyn_v1
sim.scheduler_data['dt_sigma'] = 1.*60.

if part==1:
    for ti in range(1,tn):
        sim.run_scan(t=dt)
        sim.load_detections()
        sim.sim_status(print_to='run_scan-step-%i'%(ti))

        sim.scheduler_data['t0'] = dt*(ti-1)
        sim.scheduler_data['t1'] = dt*ti

        sim.schedule_tracking()
        sim.load_schedule()
        sim.maintain_discovered()

if part==2:
    sim.load_schedule()
    sim.maint_statistics(save=True)
    sim.scan_statistics(save=True)
    sim.track_statistics(save=True,version='_cont_sched')
    sim.detection_statistics(save=True,version='_cont_sched')
    sim.maintenance_statistics(save=True,version='_cont_sched')
    sim.sim_status()


if part==3.1:
    sim.load_schedule()
    sim.generate_priors()
    sim.sim_status()

if part==3.2:
    sim.load_schedule()
    sim.generate_tracklets()
    sim.sim_status()
