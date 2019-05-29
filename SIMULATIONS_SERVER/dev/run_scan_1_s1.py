#!/usr/bin/env python
#
#
from sim_config_1_s1 import sim
import scheduler_library as sch
#import pdb

sim.scheduler = sch.dynamic_sceduler
part = 1
ti = 1
dt = 6.0

if part==0:
    sim.clear_simulation()

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
    sim.scheduler_data['que_function'] = sch.que_value_dyn_v1
    sim.scheduler_data['dt_sigma'] = 60*3 #pm 3 minutes
    sim.scheduler_data['N_sigma'] = 30
    sim.schedule_tracking()

    sim.print_tracks()
    sim.track_statistics(save=True,version='_dynamic_sceduler_ti%i'%(ti,))
    #sim.detection_statistics(save=True,version='_dynamic_sceduler_ti%i'%(ti,))
    #sim.maintenance_statistics(save=True,version='_dynamic_sceduler_ti%i'%(ti,))

if part==3.8: #security aproach
    from antenna import full_gain2inst_gain
    sim.load_schedule()
    track_n = 0
    track_rem = 0
    snr_lim = 10.0
    for track,t in sim.tracks():
        new_gain = full_gain2inst_gain(gain =  track[3],groups = 54,N_IPP = sim._obs_data.N_IPP,IPP_scale=1.0)
        print('Old gain: %.2f dB , New gain: %.2f dB ' %(track[3],new_gain))
        track_n+=1
        if new_gain >= snr_lim:
            track[2] = False
            track_rem+=1
    sim.print_tracks()
    print('Total number of tracks: %i, SNR limit at subgroup %.2f dB removed %i tracklets and %i tracklets left (%.2f %%)'% \
        (track_n,snr_lim,track_rem,track_n-track_rem,float(track_n-track_rem)/float(track_n)*100.0) )

if part==3.5:
    sim.load_schedule()
    sim.print_tracks()