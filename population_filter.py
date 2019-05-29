#!/usr/bin/env python

'''Investigate what fraction of objects can be detected with a radar system.

At least two somewhat straightforward definitions can be made.
We'll use :math:`D_{24h}` due to it being very simple to evaluate numerically.

 :math:`D_{24h}`, Can the object be detected in 24 hours of observations.
 - Makes more sense in terms of objects that can be maintained in a catalog.
   It an object is observed less than once a day, then it probably cannot be
   maintained in a catalog very well. 
 
 :math:`D_{\infty}`, Can the object be detected in infinite number of days.
 - This could be analytically determined, by using info on eccentricity, apogee, inclination,
   and object size. Not sure why this would be more useful than :math:`D_{24h}`
'''


import numpy as n
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
import time
import h5py

import simulate_tracking
import coord
import debris


def get_passes_simple(o,radar,t0,t1,max_dpos=100e3,debug=False, sanity_check=False):
    '''Follow object and find peak SNR. Assume that this occurs for minimum zenith angle of each TX.

    :param SpaceObject o: The object in space to be followed.
    :param RadarSystem radar: The radar system used for tracking.
    :param float t0: Start time for tracking
    :param float t1: End time for tracking
    :param float max_dpos: Maximum separation in m between orbital evaluation points, used to calculate time-step size by approximating orbits as circles
    :param bool debug: Verbose output
    :param bool sanity_check: Even more verbose output
    :return: Tuple of (Peak SNR for tracking, Number of receivers that can observe, time of best detection)
    '''


    # figure out the number of time points we need to evaluate to meet the max_dpos criteria
    num_t = simulate_tracking.find_linspace_num(t0, t1, o.a*1e3, o.e, max_dpos=max_dpos)
    
    # time vector
    t=n.linspace(t0,t1,num=num_t)
    dt = (t1-t0)/num_t

    # propagate object for all time points requested
    ecef=o.get_orbit(t)

    # zenith angles
    zenith = []
    
    # tx and rx site locations in ecef
    tx_ecef = []
    rx_ecef = []
    # zenith directions for each tx and rx site
    zenith_tx = []
    zenith_tx = []    
    for tx in radar._tx:
        tx_ecef.append( tx.ecef )
        zenith_tx.append( coord.azel_ecef(tx.lat, tx.lon, 0.0, 0.0, 90.0) )

    zenith_rx = []
    for rx in radar._rx:
        rx_ecef.append( rx.ecef )
        zenith_rx.append( coord.azel_ecef(rx.lat, rx.lon, 0.0, 0.0, 90.0) )
    
    # position vectors between tx and rx
    pos_rx = []
    for rxp0 in rx_ecef:
        pos_vec=(ecef.T-rxp0).T
        pos_rx.append(pos_vec)

    n_tx=len(radar._tx)
    n_rx=len(radar._rx)

    # peak snr for tx->rx combo
    peak_snr=n.zeros([n_tx,n_rx])
    # number of receivers that can observe TX
    n_rx_detections=n.zeros(n_tx)

    # for each transmitter
    for txi,txp0 in enumerate(tx_ecef):
        # tx 
        tx = radar._tx[txi]
        # zenith direction vector for this TX
        zenith = zenith_tx[txi]
        # position vector
        pos_tx=(ecef.T-txp0).T
        # unit vector for tx->target position vector
        pos_tx0=pos_tx/n.sqrt(pos_tx[0,:]**2.0+pos_tx[1,:]**2.0+pos_tx[2,:]**2.0)
        
        # zenith -> target angle
        z_angles_tx=180.0*n.arccos(pos_tx0[0,:]*zenith[0]+pos_tx0[1,:]*zenith[1]+pos_tx0[2,:]*zenith[2])/n.pi

        # peak elevation angle
        min_z=n.min(z_angles_tx)

        det_idx=n.argmin(z_angles_tx)        
        if min_z < (90.0-radar._tx[txi].el_thresh):
            # object possibly detectable
            pos_vec=pos_rx[txi][:,det_idx]
            tx_dist=n.linalg.norm(pos_vec)

            # point tx antenna towards target
            k0 = tx.point_ecef(pos_vec)
            gain_tx = tx.beam.gain(k0)
            # for all receivers
            for rxi,rx in enumerate(radar._rx):
                # position vector
                pos_rx_now=pos_rx[rxi][:,det_idx]
                # distance
                rx_dist=n.linalg.norm(pos_rx_now)
                # unit vector
                pos_rx_now0=pos_rx_now/rx_dist

                if sanity_check:
                    pos = radar._rx[rxi].ecef + pos_rx_now0*rx_dist
                    print("diff %d"%(rxi))
                    print((ecef[:,det_idx] - pos))
                
                zenith = zenith_rx[rxi]
                # rx zenith -> target angle
                z_angle_rx=180.0*n.arccos(pos_rx_now0[0]*zenith[0]+pos_rx_now0[1]*zenith[1]+pos_rx_now0[2]*zenith[2])/n.pi

                # point towards object
                k0 = rx.point_ecef(pos_rx_now)
                
                gain_rx = rx.beam.gain(k0)
                
                snr=debris.hard_target_enr(gain_tx,
                               gain_rx,
                               rx.wavelength,
                               tx.tx_power,
                               tx_dist,
                               rx_dist,
                               diameter_m=o.diam,
                               bandwidth=tx.coh_int_bandwidth,
                               rx_noise_temp=rx.rx_noise)

                peak_snr[txi,rxi]=snr
                if snr >= tx.enr_thresh:
                    n_rx_detections[txi]+=1.0
                    print("oid %d inc %1.2f diam %1.2f tx %d rx %d snr %1.2g min_range %1.2f (km) tx_dist %1.2f (km) rx_dist %1.2f (km) tx_zenith angle %1.2f rx zenith angle %1.2f"%(o.oid,o.i,o.diam,txi,rxi,snr,((1.0-o.e)*o.a)-6371.0,tx_dist/1e3,rx_dist/1e3,min_z,z_angle_rx))
                else:
                    print("oid {} not detected, SNR = {}".format(o.oid, snr))
    return peak_snr, n_rx_detections, t[det_idx]


def _save_result(detectable,n_rx,peak_snr,fname="master/eiscat3d_filter.h5"):
    '''Save results from filtering to hdf5 file.
    '''
    ho=h5py.File(fname,"w")

    ho["detectable"]= detectable > 0.5 #save as bool array
    ho["n_rx"]=n_rx
    ho["peak_snr"]=peak_snr
    ho.close()


def filter_objects(radar, m, ofname="det_filter.h5", prop_time = 24.0):
    '''Propagate for a number of hours, and determine if radar system can detect the object.
    
    :param RadarSystem radar: The radar configuration used for the detectability filtering.
    :param Population m: Input population to filter.
    :param str ofname: Output file name. If :code:`None` then the results are returned by the function instead of written to file.
    :param float prop_time: Time to propagate when filtering.
    
    # TODO: Shouldent this function use the radar snr treshold to and not only enr treshhold of antennas?
    '''
    n_total=0.0
    n_iterated=0.0
    t0=time.time()
    
    # is object detectable
    detectable=n.zeros(len(m))
    # how many tx and rx
    n_tx=len(radar._tx)
    n_rx=len(radar._rx)
    # number of rx that detect target for each tx
    n_rx_dets=n.zeros([len(m),n_tx])
    
    # peak snr on each tx->rx combo
    peak_snrs=n.zeros([len(m),n_tx,n_rx])
    
    oids = n.arange(len(m))

    inds_vector = range(MPI.COMM_WORLD.rank,len(oids),MPI.COMM_WORLD.size)
    for ID in inds_vector:
        oid = oids[ID]
        n_iterated+=1.0
        o=m.get_object(oid)
        # 24 hours from epoch
        peak_snr, n_rx_det, det_t=get_passes_simple(o,radar,0,prop_time*3600,max_dpos=100.0e3)
        #print(peak_snr,det_t)
        #print(res["snr"])
        if n.max(n_rx_det) > 0:
            for txi in range(n_tx):
                n_rx_dets[oid,txi]=n_rx_det[txi]
                for rxi in range(n_rx):                    
                    peak_snrs[oid,txi,rxi]=peak_snr[txi,rxi]
            detectable[oid]=1.0
            n_total+=1.0
                
        t1=time.time()

        print("\nThread %i, object %i done: Wall clock time to process catalog %.2f h (%i of %i) objects done, estimated time left %.2f h, fraction detectable %.2f %% \n"%( \
            MPI.COMM_WORLD.rank, oid, len(inds_vector)*(t1-t0)/float(n_iterated)/3600.0, int(n_iterated),len(inds_vector),\
             (1.0 - float(n_total)/float(len(inds_vector)) )*len(inds_vector)*(t1-t0)/float(n_iterated)/3600.0 ,\
             float(n_total)/float(len(inds_vector))*100.0) )

        if ofname is not None and MPI.COMM_WORLD.size == 1:
            if int(n_total) % 100 == 0:
                print("saving")
                _save_result(detectable,n_rx_dets,peak_snrs,fname=ofname)

    print('Threads: MPI.COMM_WORLD.size = %i'%(MPI.COMM_WORLD.size))
    if MPI.COMM_WORLD.size > 1:
        
        if MPI.COMM_WORLD.rank == 0:
            print('---> Thread %i: Reciving all filter results <barrier>'%(MPI.COMM_WORLD.rank))
            for T in range(1,MPI.COMM_WORLD.size):
                for ID in range(T,len(oids),MPI.COMM_WORLD.size):
                    oid = oids[ID]
                    detectable[oid] = MPI.COMM_WORLD.recv(source=T, tag=oid+1)
                    n_rx_dets[oid]  = MPI.COMM_WORLD.recv(source=T, tag=oid+2)
                    peak_snrs[oid]  = MPI.COMM_WORLD.recv(source=T, tag=oid+3)
        else:
            print('---> Thread %i: Distributing all filter results to thread 0 <barrier>'%(MPI.COMM_WORLD.rank))
            for oid in oids[inds_vector]:
                MPI.COMM_WORLD.send(detectable[oid], dest=0, tag=oid+1)
                MPI.COMM_WORLD.send(n_rx_dets[oid],  dest=0, tag=oid+2)
                MPI.COMM_WORLD.send(peak_snrs[oid],  dest=0, tag=oid+3)
        print('---> Distributing done </barrier>')

    if ofname is not None:
        if MPI.COMM_WORLD.rank == 0:
            print("---> saving_final")
            _save_result(detectable,n_rx_dets,peak_snrs,fname=ofname)
        
        if MPI.COMM_WORLD.size > 1: #wait for thread 0 saving so that we dont run off and do something before that
            print('---> Post-save <barrier>')
            MPI.COMM_WORLD.barrier()
            print('---> Saving done </barrier>')
    else:
        if MPI.COMM_WORLD.size > 1:
            detectable = MPI.COMM_WORLD.bcast(detectable, root=0)
            n_rx_dets = MPI.COMM_WORLD.bcast(n_rx_dets, root=0)
            peak_snrs = MPI.COMM_WORLD.bcast(peak_snrs, root=0)

        return detectable, n_rx_dets, peak_snrs
        

        
if __name__ == "__main__":
    import population_library as plib 
    import radar_library as rlib

    mc = plib.master_catalog()

    mc.filter('i', lambda x: x >= 50.0)
    mc.filter('d', lambda x: x >= 1.0)

    mc.delete(slice(20,None,1))

    print('Population size: {}'.format(len(mc)))

    radars=[
        rlib.eiscat_3d(),
        #rlib.eiscat_3d_module(),
    ]

    for radar in radars:
        ofname = "master/{}_filter.h5".format(radar.name.replace(' ', '_'))
        detectable, n_rx_dets, peak_snrs = filter_objects(radar, mc, ofname=None, prop_time=24.0)
        print(detectable)
        print(n_rx_dets)
        print(peak_snrs)



