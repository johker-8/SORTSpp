import numpy as n
import matplotlib.pyplot as plt
import h5py
import time

# SORTS imports
import space_object as so
from simulate_scan import get_iods
import radar_library as rl
import radar_scan_library as rslib
import antenna_library as alib

e3d = rl.eiscat_3d()

e3d.set_FOV(max_on_axis=25.0,horizon_elevation=30.0)
e3d.set_SNR_limits(min_total_SNRdb=10.0,min_pair_SNRdb=0.0)
e3d.set_TX_bandwith(bw = 1.0e6)

#initialize the observing mode
e3d_scan = rslib.ns_fence_rng_model(min_el = 30.0, angle_step = 2.0, dwell_time = 0.1)

e3d_scan.set_radar_location(e3d)
e3d.set_scan(e3d_scan)

#setup space object
o=so.space_object(a=7000,e=0.0,i=72,raan=0,aop=0,mu0=0,C_D=2.3,A=1.0,m=1.0,diam=0.1)


ITER = 1
dt = 24

e3d.set_beam('TX', alib.cassegrain_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.3,a0=80.0, a1=80.0/16.0*2.29) )
e3d.set_beam('RX', alib.cassegrain_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.3,a0=80.0, a1=80.0/16.0*2.29) )

exec_time_cassegrain = n.zeros((ITER,))
for I in range(ITER):
    t0 = time.time()
    # get all IODs for one object during 24 hours
    det_times=get_iods(o,e3d,0,dt*3600)
    t1=time.time()
    exec_time_cassegrain[I] = (t1-t0)/60.0
    print("Wall clock time %1.2f min: cassegrain"%( exec_time_cassegrain[I], ))
    print('Detections:',det_times)
exec_time_cassegrain_mean = n.mean(exec_time_cassegrain)
print("Wall clock time full array MEAN: %1.2f min"%( exec_time_cassegrain_mean, ))



e3d.set_beam('TX', alib.e3d_array_beam_stage1(opt='dense') )
e3d.set_beam('RX', alib.e3d_array_beam() )

exec_time_array = n.zeros((ITER,))
for I in range(ITER):
	t0 = time.time()
	# get all IODs for one object during 24 hours
	det_times=get_iods(o,e3d,0,dt*3600)
	t1=time.time()
	exec_time_array[I] = (t1-t0)/60.0
	print("Wall clock time %1.2f min: full array"%( exec_time_array[I], ))
	print('Detections:',det_times)
exec_time_array_mean = n.mean(exec_time_array)
print("Wall clock time full array MEAN: %1.2f min"%( exec_time_array_mean, ))



print('\n\n Percentage exec time increase: %.4f %%'%( exec_time_array_mean/exec_time_cassegrain_mean*100.-100. ))