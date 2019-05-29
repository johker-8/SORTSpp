#!/usr/bin/env python
#
#
from sim_maint_sst_gbeam_e3d import sim
import scheduler_library as sch

sim.scheduler = sch.dynamic_sceduler
part = 3
ti = 1
dt = 1.0

if part==1:
    sim._catalogue.maintain_all()
    sim.run_scan(t=dt)
    sim.load_detections(par=True) #to save
    sim.sim_status(print_to='run_scan-step-%i'%(ti))
    sim.maint_statistics(save=True)
    sim.scan_statistics(save=True)

if part==1.5:
    sim.load_detections()
    sim.maint_statistics(save=True)
    sim.scan_statistics(save=True)

if part==2:
    sim.load_detections()
    sim.scheduler_data['t0'] = dt*(ti-1)
    sim.scheduler_data['t1'] = dt*ti
    sim.scheduler_data['que_function'] = sch.que_value_dyn_v2
    
    sim.scheduler_data['N_on'] = True
    sim.scheduler_data['dt_on'] = True
    sim.scheduler_data['tracklets_on'] = True
    sim.scheduler_data['peak_snr_on'] = True
    sim.scheduler_data['source_on'] = True

    sim.scheduler_data['N_rate'] = 50.
    sim.scheduler_data['dt_sigma'] = 5*1.*60.
    sim.scheduler_data['dt_offset'] = -1.*60. #shifts to later than max
    sim.scheduler_data['dt_sqew'] = -0.8 # > 0 = initial trail longer
    sim.scheduler_data['tracklets_scale'] = 15.
    sim.scheduler_data['peak_snr_rate'] = 50.
    sim.scheduler_data['track-scan_ratio'] = 0.5

    sim.scheduler_data['N_scale'] = 1.
    sim.scheduler_data['dt_scale'] = 5.
    sim.scheduler_data['tracklets_rate'] = 2.
    sim.scheduler_data['peak_snr_scale'] = 0.5
    sim.scheduler_data['tracklet_completion_rate'] = 20.0

    sim.schedule_tracking()

    sim.print_tracks()
    sim.track_statistics(save=True,version='_dynamic_sceduler_ti%i'%(ti,))
    sim.detection_statistics(save=True,version='_dynamic_sceduler_ti%i'%(ti,))
    sim.maintenance_statistics(save=True,version='_dynamic_sceduler_ti%i'%(ti,))

if part==3:
    sim.load_schedule()
    sim.sim_status()
#    sim.generate_priors()
    sim.generate_tracklets()
