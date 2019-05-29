#!/usr/bin/env python

'''
Estimates a space objects state vector from a set of ranges and range-rates.

'''

import scipy.optimize as so
import numpy as n

import glob

import h5py
import re

debug_low=False

# simple ecef tri-static position estimate 
# using fmin search
# p_rx is receiver locations
def meas2pos(m,p_rx):
    model=n.zeros(3)
    def ss(x):
        # try position
        p=n.array(x)

        # distance from tx to target
        tx2p=n.linalg.norm(p-p_rx[:,0])

        # distance from tx to target and rx 1 tx-target-tx
        model[0]=tx2p*2.0
        # distance from tx to target and to rx 2
        model[1]=tx2p+n.linalg.norm(p-p_rx[:,1])
        # distance from tx to target and to rx 3        
        model[2]=tx2p+n.linalg.norm(p-p_rx[:,2])
        
        alt=n.linalg.norm(p)-6370.0

        if alt < 0.0:
            return(1e99)
        ssq=n.sum(n.abs(model-m)**2.0)
        return(ssq)
    
    xhat=so.fmin(ss,n.array([10000,10000,10000]),full_output=False,disp=False)
    if debug_low:
        print("residual %1.2f"%(ss(xhat)))
    return(xhat)

# simple ecef tri-static velocity vector estimate 
# using fmin search
# p_rx is receiver locations
# dt is numerical derivative length
def meas2vel(p,m,p_rx,dt=0.1):
    model=n.zeros(3)
    def ss(x):
        v=n.array(x[0:3])
        # delay between pos and rx 0
        delay0=n.linalg.norm(p-p_rx[:,0])
        # delay between pos and rx 1        
        delay1=n.linalg.norm(p-p_rx[:,1])
        # delay between pos and rx 2                
        delay2=n.linalg.norm(p-p_rx[:,2])
        
        # target-rx0 delay after dt
        delay0_dt=n.linalg.norm(p+v*dt-p_rx[:,0]) 
        model[0]=(2*delay0_dt-2*delay0)/dt

        # target-rx1 delay after dt
        delay1_dt=n.linalg.norm(p+v*dt-p_rx[:,1]) 
        model[1]=( (delay0_dt+delay1_dt)-(delay0+delay1) )/dt

        # target-rx2 delay after dt
        delay2_dt=n.linalg.norm(p+v*dt-p_rx[:,2])         
        model[2]=( (delay0_dt+delay2_dt)-(delay0+delay2) )/dt
        
        ssq=n.sum(n.abs(model-m)**2.0)
        return(ssq)
    
    xhat=so.fmin(ss,n.array([1,1,1]),full_output=False,disp=False)
    if debug_low:
        print("residual %1.2f"%(ss(xhat)))    
    return(xhat)

# estimate full state vector using range and range-rate measurements
# p_rx is receiver locations. first one is transmit position in ecef
def estimate_state(r_meas,rr_meas,p_rx):
    pos=meas2pos(r_meas,p_rx)
    vel=meas2vel(pos,rr_meas,p_rx)
    ecef_state=n.zeros(6)
    ecef_state[0:3]=pos
    ecef_state[3:6]=vel
    return(ecef_state)


def state_estimation(tracklet_list, verbose = False):

    ecef_states = []
    t = []

    if len(tracklet_list) < 3:
        print("ERROR, non tri-static tracklets, skipping")
        return ecef_states,t
    if verbose:
        for fname in tracklet_list:
            print("Track %s"%fname)
    #h5_tr_list = [ h5py.File(fname,"r") for fname in tracklet_list ]
    h0=h5py.File(tracklet_list[0],"r")
    h1=h5py.File(tracklet_list[1],"r")
    h2=h5py.File(tracklet_list[2],"r") 

    t_means0=h0["m_time"].value

    r_meas0=h0["m_range"].value
    rr_meas0=h0["m_range_rate"].value    
    r_meas1=h1["m_range"].value
    rr_meas1=h1["m_range_rate"].value
    r_meas2=h2["m_range"].value
    rr_meas2=h2["m_range_rate"].value

    n_t=len(r_meas0)
    if len(r_meas1) != n_t or len(r_meas2) != n_t:
        print("non-overlapping measurements, tbd, align measurement")
        return ecef_states,t
    
    p_rx=n.zeros([3,3])
    p_rx[:,0]=h0["rx_loc"].value/1e3
    p_rx[:,1]=h1["rx_loc"].value/1e3
    p_rx[:,2]=h2["rx_loc"].value/1e3    

    ecef_states.append(n.zeros((6,n_t)))
    t.append(n.zeros((1,n_t)))
    for ti in range(n_t):
        if h0["m_time"][ti] != h1["m_time"][ti] or h2["m_time"][ti] != h0["m_time"][ti]:
            print("non-aligned measurement")
            continue
        m_r=n.array([r_meas0[ti],r_meas1[ti],r_meas2[ti]])
        m_rr=n.array([rr_meas0[ti],rr_meas1[ti],rr_meas2[ti]])
        ecef_state=estimate_state(m_r,m_rr,p_rx)
        true_state=h0["true_state"].value[ti,:]
        
        ecef_states[-1][:,ti] = ecef_state
        t[-1][0,ti] = t_means0[ti]
        
        if verbose:
            print("mean pos error %1.2f (km) mean vel error %1.2f (km)"%(n.mean(ecef_state[0:3]-true_state[0:3]),n.mean(ecef_state[3:6]-true_state[3:6])))
    h0.close()
    h1.close()
    h2.close()
    return ecef_states,t


def state_estimation_v2(tracklet_folder, track_id = -1, verbose = False):
    fl2=glob.glob("%s/*.h5"%(tracklet_folder))
    fl2.sort()
    
    #parse files
    file_id_list = []
    n_base_tracks = []
    n_track_file_start = []
    current_track = ''
    for i in range(len(fl2)):
        file_id_list.append(fl2[i][-26:-3].split('-'))

        if current_track == file_id_list[i][0] and i !=0:
            n_base_tracks[-1] += 1
        else:
            current_track = file_id_list[i][0]
            if n.sum(n_base_tracks) == 0.0:
                n_track_file_start.append(0)
            else:
                n_track_file_start.append(0 + n.sum(n_base_tracks))
            n_base_tracks.append( 1 ) #track id at pos, contains number of track baselines
    n_tracks = len(n_base_tracks)

    ecef_states = []
    t = []

    #print(n_track_file_start)

    if track_id >= 0:
        if track_id < n_tracks:
            compute_list = [track_id]
        else:
            print('ERROR, track id does not exist')
            return ecef_states,t
    else:
        compute_list = range(n_tracks)

    for i in compute_list:
        if n_base_tracks[i] < 3:
            print("ERROR, non tri-static tracklets, skipping")
            continue
        if verbose:
            print("Track %d"%(i))
            print(fl2[0 + n_track_file_start[i]])
            print(fl2[1 + n_track_file_start[i]])
            print(fl2[2 + n_track_file_start[i]])            
        h0=h5py.File(fl2[0 + n_track_file_start[i]],"r")
        h1=h5py.File(fl2[1 + n_track_file_start[i]],"r")
        h2=h5py.File(fl2[2 + n_track_file_start[i]],"r") 

        t_means0=h0["m_time"].value

        r_meas0=h0["m_range"].value
        rr_meas0=h0["m_range_rate"].value    
        r_meas1=h1["m_range"].value
        rr_meas1=h1["m_range_rate"].value
        r_meas2=h2["m_range"].value
        rr_meas2=h2["m_range_rate"].value

        n_t=len(r_meas0)
        if len(r_meas1) != n_t or len(r_meas2) != n_t:
            print("non-overlapping measurements, tbd, align measurement")
            continue
        
        p_rx=n.zeros([3,3])
        p_rx[:,0]=h0["rx_loc"].value/1e3
        p_rx[:,1]=h1["rx_loc"].value/1e3
        p_rx[:,2]=h2["rx_loc"].value/1e3    

        ecef_states.append(n.zeros((6,n_t)))
        t.append(n.zeros((1,n_t)))
        for ti in range(n_t):
            if h0["m_time"][ti] != h1["m_time"][ti] or h2["m_time"][ti] != h0["m_time"][ti]:
                print("non-aligned measurement")
                continue
            m_r=n.array([r_meas0[ti],r_meas1[ti],r_meas2[ti]])
            m_rr=n.array([rr_meas0[ti],rr_meas1[ti],rr_meas2[ti]])
            ecef_state=estimate_state(m_r,m_rr,p_rx)
            true_state=h0["true_state"].value[ti,:]
            
            ecef_states[-1][:,ti] = ecef_state
            t[-1][0,ti] = t_means0[ti]
            
            if verbose:
                print("mean pos error %1.2f (km) mean vel error %1.2f (km)"%(n.mean(ecef_state[0:3]-true_state[0:3]),n.mean(ecef_state[3:6]-true_state[3:6])))
        h0.close()
        h1.close()
        h2.close()
    return ecef_states,t


def test_tracklets():
    # Unit test
    #
    # Create tracklets and perform orbit determination
    #
    import population_library as plib
    import radar_library as rlib
    import simulate_tracking
    import simulate_tracklet as st
    import os

    from propagator_neptune import PropagatorNeptune
    from propagator_sgp4 import PropagatorSGP4
    from propagator_orekit import PropagatorOrekit

    opts_sgp4 = dict(
        polar_motion = False,
        out_frame = 'ITRF',
    )

    opts_nept = dict()
    opts_orekit = dict(
        in_frame='TEME',
        out_frame='ITRF',
    )


    os.system("rm -Rf /tmp/test_tracklets")
    os.system("mkdir /tmp/test_tracklets")    
    m = plib.master_catalog(
        sort=False,
        propagator = PropagatorNeptune,
        propagator_options = opts_nept,
    )

    # Envisat
    o = m.get_object(145128)
    print(o)

    e3d = rlib.eiscat_3d(beam='gauss')

    # time in seconds after mjd0
    t_all = n.linspace(0, 24*3600, num=1000)
    
    passes, _, _, _, _ = simulate_tracking.find_pass_interval(t_all, o, e3d)
    print(passes)

    for p in passes[0]:
        # 100 observations of each pass
        mean_t=0.5*(p[1]+p[0])
        print("duration %1.2f"%(p[1]-p[0]))
        if p[1]-p[0] > 10.0:
            t_obs=n.linspace(mean_t-10,mean_t+10,num=10)
            print(t_obs)
            meas, fnames, ecef_stdevs = st.create_tracklet(o, e3d, t_obs, hdf5_out=True, ccsds_out=True, dname="/tmp/test_tracklets")
    
    fl=glob.glob("/tmp/test_tracklets/*")
    for f in fl:
        print(f)
        fl2=glob.glob("%s/*.h5"%(f))
        print(fl2)
        fl2.sort()
        start_times=[]
        for f2 in fl2:
            start_times.append(re.search("(.*/track-.*)-._..h5",f2).group(1))
        start_times=n.unique(start_times)
        print("n_tracks %d"%(len(start_times)))
        
        for t_pref in start_times:
            fl2 = glob.glob("%s*.h5"%(t_pref))
            n_static=len(fl2)
            if n_static == 3:
                print("Fitting track %s"%(t_pref))

                f0="%s-0_0.h5"%(t_pref)
                f1="%s-0_1.h5"%(t_pref)
                f2="%s-0_2.h5"%(t_pref)                

                print(f0)
                print(f1)
                print(f2)                
                
                h0=h5py.File(f0,"r")
                h1=h5py.File(f1,"r")
                h2=h5py.File(f2,"r") 
                
                r_meas0=h0["m_range"].value
                rr_meas0=h0["m_range_rate"].value    
                r_meas1=h1["m_range"].value
                rr_meas1=h1["m_range_rate"].value    
                r_meas2=h2["m_range"].value
                rr_meas2=h2["m_range_rate"].value

                n_t=len(r_meas0)
                if len(r_meas1) != n_t or len(r_meas2) != n_t:
                    print("non-overlapping measurements, tbd, align measurement")
                    continue
            
                p_rx=n.zeros([3,3])
                p_rx[:,0]=h0["rx_loc"].value/1e3
                p_rx[:,1]=h1["rx_loc"].value/1e3
                p_rx[:,2]=h2["rx_loc"].value/1e3    
    
                for ti in range(n_t):
                    if h0["m_time"][ti] != h1["m_time"][ti] or h2["m_time"][ti] != h0["m_time"][ti]:
                        print("non-aligned measurement")
                        continue
                    m_r=n.array([r_meas0[ti],r_meas1[ti],r_meas2[ti]])
                    m_rr=n.array([rr_meas0[ti],rr_meas1[ti],rr_meas2[ti]])
                    ecef_state=estimate_state(m_r,m_rr,p_rx)
                    true_state=h0["true_state"].value[ti,:]
                    print("pos error %1.3f (m) vel error %1.3f (m/s)"%(1e3*n.linalg.norm(ecef_state[0:3]-true_state[0:3]),1e3*n.linalg.norm(ecef_state[3:6]-true_state[3:6])))
                h0.close()
                h1.close()
                h2.close()


def neptune_state_est(dname="./neptune_tracklets/E3D_neptune/scheduler_2d/tracklets/1001240430178"):
    
    
    # find measurement triplets and analyze them
    fl=glob.glob("%s"%(dname))
    for f in fl:
        print(f)
        fl2=glob.glob("%s/*.h5"%(f))
        print(fl2)
        fl2.sort()
        start_times=[]
        for f2 in fl2:
            start_times.append(re.search("(.*/track-.*)-._..h5",f2).group(1))
        start_times=n.unique(start_times)
        print("n_tracks %d"%(len(start_times)))
        
        for t_pref in start_times:
            fl2 = glob.glob("%s*.h5"%(t_pref))
            n_static=len(fl2)
            if n_static == 3:
                print("Fitting track %s"%(t_pref))

                f0="%s-0_0.h5"%(t_pref)
                f1="%s-0_1.h5"%(t_pref)
                f2="%s-0_2.h5"%(t_pref)                

                print(f0)
                print(f1)
                print(f2)                
                
                h0=h5py.File(f0,"r")
                h1=h5py.File(f1,"r")
                h2=h5py.File(f2,"r") 
                
                r_meas0=h0["m_range"].value
                rr_meas0=h0["m_range_rate"].value    
                r_meas1=h1["m_range"].value
                rr_meas1=h1["m_range_rate"].value    
                r_meas2=h2["m_range"].value
                rr_meas2=h2["m_range_rate"].value

                n_t=len(r_meas0)
                if len(r_meas1) != n_t or len(r_meas2) != n_t:
                    print("non-overlapping measurements, tbd, align measurement")
                    continue
            
                p_rx=n.zeros([3,3])
                p_rx[:,0]=h0["rx_loc"].value/1e3
                p_rx[:,1]=h1["rx_loc"].value/1e3
                p_rx[:,2]=h2["rx_loc"].value/1e3    
    
                for ti in range(n_t):
                    if h0["m_time"][ti] != h1["m_time"][ti] or h2["m_time"][ti] != h0["m_time"][ti]:
                        print("non-aligned measurement")
                        continue
                    m_r=n.array([r_meas0[ti],r_meas1[ti],r_meas2[ti]])
                    m_rr=n.array([rr_meas0[ti],rr_meas1[ti],rr_meas2[ti]])
                    ecef_state=estimate_state(m_r,m_rr,p_rx)
                    true_state=h0["true_state"].value[ti,:]

                    print("pos error %1.3f (m) vel error %1.3f (m/s)"%(1e3*n.linalg.norm(ecef_state[0:3]-true_state[0:3]),1e3*n.linalg.norm(ecef_state[3:6]-true_state[3:6])))
                h0.close()
                h1.close()
                h2.close()

 
if __name__ == "__main__":
    test_tracklets()
#    neptune_state_est(dname="./neptune_tracklets/E3D_neptune/scheduler_2d/tracklets/1001013500019")

