#!/usr/bin/env python
#
#
import numpy as n
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt
#scaning snr curve

#SORTS Libraries
import radar_library as rl
import radar_scan_library as rslib
import scheduler_library as sch
import antenna_library as alib
import simulate_tracking as st
import space_object as so
from antenna import full_gain2inst_gain,inst_gain2full_gain
from simulate_scaning_snr import simulate_full_scaning_snr_curve
import antenna

if __name__ == "__main__":
    #initialize the radar setup
    radar = rl.eiscat_3d()

    radar.set_FOV(max_on_axis=25.0,horizon_elevation=30.0)
    radar.set_SNR_limits(min_total_SNRdb=1.0,min_pair_SNRdb=0.0)
    radar.set_TX_bandwith(bw = 1.0e6)

    #tx_beam = alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.2,a0=40.0,az1=0,el1=90.0)
    tx_beam = alib.e3d_array_beam_stage1(opt='dense')
    #antenna.plot_gain(tx_beam,res=300,min_el=80.0)

    radar.set_beam('TX', tx_beam )
    #radar.set_beam('RX', alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.5,a0=40.0,az1=0,el1=90.0) )
    radar.set_beam('RX', alib.e3d_array_beam() )

    #initialize the observing mode
    #radar_scan = rslib.ns_fence_rng_model(min_el = 30.0, angle_step = 2.0, dwell_time = 0.1)
    
    #az_points = n.arange(0,360,45).tolist() + [0.0];
    #el_points = [90.0-n.arctan(50.0/300.0)*180.0/n.pi, 90.0-n.arctan(n.sqrt(2)*50.0/300.0)*180.0/n.pi]*4+[90.0];
    #radar_scan = rslib.n_const_pointing_model(az_points,el_points,len(az_points),dwell_time = 0.4)
    radar_scan = rslib.beampark_model(az=0.0,el=90.0)

    radar_scan.set_radar_location(radar)
    radar.set_scan(radar_scan)

    obs_par = sch.calculate_observation_params( \
        duty_cycle = 0.25, \
        SST_f = 1.0, \
        tracking_f = 0.0, \
        coher_int_t=0.2)
    sch.configure_radar_to_observation(radar,obs_par,'scan')

    # get all IODs for one object during 24 hours
    o=so.space_object(a=7000,e=0.0,i=72,raan=0,aop=0,mu0=0,C_D=2.3,A=1.0,m=1.0,diam=0.1)
            
    det_times=st.get_passes(o,radar,69*3600.,70*3600.,max_dpos=50.0)
    #['t'][tx 0][pass 2][above horizon time = 0, below = 1]

    ts,angs = st.get_angles(det_times,o,radar)

    #st.plot_angles(ts,angs)
    st.print_passes(det_times)

    tresh = 10.0 #DETECTION LIMIT IN COHERRENTLY INTEGRATED SNR
    rem_t = 10.0 #HARD REMOVAL LIMIT IN SUBGROUP SNR

    simulate_full_scaning_snr_curve(radar,o,det_times,tresh,rem_t,obs_par,groups=54, plot = True, verbose=True)
