import numpy as n
import matplotlib.pyplot as plt

import orbit_accuracy as oa
import space_object as so
import radar_library as rl
import antenna_library as alib
import simulate_tracking as st
import scheduler_library as sch
import population as p

radar = rl.eiscat_3d()

radar.set_FOV(max_on_axis=25.0,horizon_elevation=30.0)
radar.set_SNR_limits(min_total_SNRdb=1.0,min_pair_SNRdb=0.0)
radar.set_TX_bandwith(bw = 1.0e6)
radar.set_beam('TX', alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.2,a0=40.0,az1=0,el1=90.0) )
radar.set_beam('RX', alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.5,a0=40.0,az1=0,el1=90.0) )
#radar.set_beam('TX', alib.e3d_array_beam_stage1(opt='dense') )
#radar.set_beam('RX', alib.e3d_array_beam() )

obs_par = sch.calculate_observation_params( \
        duty_cycle = 0.25, \
        SST_f = 1.0, \
        tracking_f = 1.0, \
        coher_int_t=0.2)
sch.configure_radar_to_observation(radar,obs_par,'track')

# space object population
m=p.master_catalog(sort=False)
# one object
#space_o=m.get_object(13)
#space_o.diam=0.1

space_o=so.space_object(a=7000,e=0.0,i=72,raan=0,aop=0,mu0=0,C_D=2.3,A=1.0,m=1.0,diam=0.1)


det_times=st.get_passes(space_o,radar,0*3600,2*3600,max_dpos=50.0, debug=False)


t_up = det_times['t'][0][0][0]
t_down = det_times['t'][0][0][1]
t_max = det_times['snr'][0][0][0][1]
t = n.linspace(t_up,t_down,num=100)
ecefs = space_o.get_state(t)

print('---> PASS %.2f h to %.2f h' %(n.min(t)/3600.0,n.max(t)/3600.0))

all_snrs = st.get_track_snr(t,space_o,radar)
all_snrs = 10.0*n.log10(n.array(all_snrs))
#st.plot_snr(t,all_snrs,radar)

tracklets = []
for txi in range(len(radar._tx)):
    tracklets.append(t.copy())

print(tracklets)

try:
    Sigma_pos,Sigma_ecef,max_var_ecef=oa.linearized_errors(space_o,radar,tracklets,plot=False,debug=False)
except n.linalg.linalg.LinAlgError as e:
    print('The orbit could not be determined within linearized errors: {}'.format(e))
    print('Skipping prior generation and moving to next pass:')
else:
    print('---> ECEF shape: ',ecefs.shape)
    print('---> The ECEF covariance matrix is:',Sigma_ecef)
    print('---> The KEP  covariance matrix is:',Sigma_pos)
    xi = n.random.multivariate_normal(n.zeros((6,)), Sigma_ecef, ecefs.shape[1]).T

    ecefs0 = ecefs.copy()
    ecefs += xi

    print(ecefs0)
    print(xi)
    print(ecefs)
