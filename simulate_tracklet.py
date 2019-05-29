#!/usr/bin/env python

'''Given scheduled observations of an object simulate the generated tracklet-data.

# TODO: Rewrite with new functionality
# TODO: Do not re-do the entire "observation simulation" as there is already modules that to this better. Instead just take a time-series in and create the tracklet, let other code worry about if it is physically correct or not.

Simulate an EISCAT 3D tracking experiment using MASTER model objects
 
 - Follow object from horizon to horizon
 - Output estimated range and range-rate errors

'''

import numpy as n
import scipy.constants as c
import matplotlib.pyplot as plt
import h5py
import os
import scipy.interpolate as sint

# SORTS imports
import coord
import debris
import plothelp
import ccsds_write
import time
import dpt_tools as dpt

debug_low=False
debug_high=False

iono_errfun = debris.ionospheric_error_fun()
'''func: Model of the ionospheric error function. See module :mod:`debris`.
'''

def write_ccsds(o, meas, tx, rx, fname):
    '''
    Write tracklet in ccsds file format

    '''
    m_time=meas["m_time"]
    m_range=meas["m_range"]
    m_range_rate=meas["m_range_rate"]
    m_range_std=meas["m_range_std"]
    m_range_rate_std=meas["m_range_rate_std"]
    
    ccsds_write.write_ccsds(m_time,
                            m_range,
                            m_range_rate,
                            m_range_std,
                            m_range_rate_std,                                    
                            tx_ecef=tx.ecef,
                            rx_ecef=rx.ecef,           
                            freq=tx.freq,
                            tx=tx.name,
                            rx=rx.name,                
                            oid="%s"%(o.oid),
                            tdm_type="track",
                            fname=fname)


def write_tracklets(o, meas, radar, dname, hdf5_out=True, ccsds_out=True, dt=3600):
    '''
    Write a tracklet files.

    '''
    fnames = []
    # for each transmitter
    for txi,tx in enumerate(radar._tx):
        # for each receiver
        for rxi,tx in enumerate(radar._rx):
            # get measurement
            omeas=meas[txi][rxi]
            tidxs=[]

            # if length of measurements is not zero
            if len(omeas["m_time"]) !=0:
                if hdf5_out:
                    os.system("mkdir -p %s/%d"%(dname,o.oid))
                    fname="%s/%d/track-%06d-%d-%d_%d.h5"%(dname,o.oid,n.min(omeas["m_time"]),o.oid,txi,rxi)
                    if debug_high:
                        print("Writing %s"%(fname))
                            
                    ho=h5py.File(fname,"w")
                    ho["oid"]=o.oid
                    
                    ho["txi"]=txi
                    ho["rxi"]=rxi                
                    
                    ho["tx_loc"]=radar._tx[txi].ecef
                    ho["rx_loc"]=radar._rx[rxi].ecef
                    
                    
                    ho["m_time"]=omeas["m_time"]
                    ho["m_range"]=omeas["m_range"]
                    ho["m_range_std"]=omeas["m_range_std"]
                    
                    ho["m_range_rate"]=omeas["m_range_rate"]
                    ho["m_range_rate_std"]=omeas["m_range_rate_std"]
                    
                    ho["g_lat"]=omeas["g_lat"]
                    ho["g_lon"]=omeas["g_lon"]
                        
                    ho["true_state"]=omeas["true_state"]
                    ho["true_time"]=omeas["true_time"]
                    ho.close()
                    
                if ccsds_out:
                    fname="%s/%d/track-%06d-%d-%d_%d.tdm"%(dname,o.oid,n.min(omeas["m_time"]),o.oid,txi,rxi)
                    if debug_high:
                        print("Writing %s"%(fname))                
                    write_ccsds(o,omeas,radar._tx[txi],radar._rx[rxi],fname)
                    fnames.append( "%s/%d/track-%06d-%d-%d_%d"%(dname,o.oid,n.min(omeas["m_time"]),o.oid,txi,rxi) )
    return fnames
    

def create_tracklet(o, radar, t_obs, hdf5_out=True, ccsds_out=True, dname="./tracklets", noise=False, dx=10.0, dv=10.0, dt = 0.01, ignore_elevation_thresh=False):
    '''Simulate tracks of objects.

    ionospheric limit is a lower limit on precision after ionospheric corrections
    '''

    if noise:
        noise=1.0
    else:
        noise=0.0

    # TDB, kludge, this should be allowed to change as a function of time
    bw=radar._tx[0].tx_bandwidth
    txlen=radar._tx[0].pulse_length*1e6 # pulse length in microseconds
    ipp=radar._tx[0].ipp # pulse length in microseconds
    n_ipp=int(radar._tx[0].n_ipp) # pulse length in microseconds
    rfun,dopfun=debris.precalculate_dr(txlen,bw,ipp=ipp,n_ipp=n_ipp)
    
    t0_unix = dpt.jd_to_unix(dpt.mjd_to_jd(o.mjd0))

    if debug_low:
        for tx in radar._tx:
            print("TX %s"%(tx.name))
        for rx in radar._tx:
            print("RX %s"%(rx.name))

    rx_p=[]
    tx_p=[]
    for tx in radar._tx:    
        tx_p.append(coord.geodetic2ecef(tx.lat, tx.lon, tx.alt))

    for rxi,rx in enumerate(radar._rx):
        rx_p.append(coord.geodetic2ecef(rx.lat, rx.lon, rx.alt))

    ecefs=o.get_orbit(t_obs)
    state=o.get_state(t_obs)    
    ecefs_p_dt=o.get_orbit(t_obs+dt)
    ecefs_m_dt=o.get_orbit(t_obs-dt)
    
    # velocity in ecef
    ecef_vel=(0.5*((ecefs_p_dt-ecefs)+(ecefs-ecefs_m_dt))/dt)

    # linearized error estimates for ecef state vector error std_dev, when three or more
    # delays and doppl er shifts are measured
    ecef_stdevs=[]
    meas=[]

    for tx in radar._tx:
        ecef_stdevs.append({"time_idx":[],
                            "m_time":[],
                            "ecef_stdev":[]})
        
    for tx in radar._tx:
        m_rx=[]
        for rx in radar._rx:
            m_rx.append({"time_idx":[],
                         "m_time":[],
                         
                         "m_delay":[],
                         "m_delay_std":[],
                         
                         "m_range":[],                                                  
                         "m_range_std":[],

                         "m_range_rate":[],
                         "m_range_rate_std":[],

                         "m_doppler":[],
                         "m_doppler_std":[],
                         
                         "gain_tx":[],
                         "gain_rx":[],

                         "enr":[],
                         
                         "true_state":[],
                         "true_time":[],
                         
                         "g_lat":[],
                         "g_lon":[]})
        meas.append(m_rx)
        
    # largest possible number of state vector measurements
    n_state_meas=len(radar._tx)*len(radar._rx)
    # jacobian for error covariance calc
    J = n.zeros([2*n_state_meas,6])
    Sigma_Lin = n.zeros([2*n_state_meas,2*n_state_meas])
    
    # error standard deviation for state vector estimates at each position
    state_vector_errors=n.zeros([6,len(t_obs)])
    
    # go through all times
    for ti,t in enumerate(t_obs):
        p=n.array([ecefs[0,ti],ecefs[1,ti],ecefs[2,ti]])
        p_p=n.array([ecefs_p_dt[0,ti],ecefs_p_dt[1,ti],ecefs_p_dt[2,ti]])
        p_m=n.array([ecefs_m_dt[0,ti],ecefs_m_dt[1,ti],ecefs_m_dt[2,ti]])

        # for linearized state vector error determination
        p_dx0=n.array([ecefs[0,ti]+dx,ecefs[1,ti],ecefs[2,ti]])
        p_dx1=n.array([ecefs[0,ti],ecefs[1,ti]+dx,ecefs[2,ti]])
        p_dx2=n.array([ecefs[0,ti],ecefs[1,ti],ecefs[2,ti]+dx])
        # doppler error comes from linear least squares
        
        # initialize jacobian
        J[:,:]=0.0
        Sigma_Lin[:,:]=0.0
        state_meas_idx=0
        
        # go through all transmitters
        for txi,tx in enumerate(radar._tx):        
            pos_vec_tx=-tx_p[txi]+p
            pos_vec_tx_p=-tx_p[txi]+p_p
            pos_vec_tx_m=-tx_p[txi]+p_m

            # for linearized errors
            pos_vec_tx_dx0=-tx_p[txi]+p_dx0
            pos_vec_tx_dx1=-tx_p[txi]+p_dx1
            pos_vec_tx_dx2=-tx_p[txi]+p_dx2
        
            # incident k-vector
            k_inc = -2.0*n.pi*pos_vec_tx/n.linalg.norm(pos_vec_tx)/tx.wavelength
            
            elevation_tx=90.0-coord.angle_deg(tx_p[txi],pos_vec_tx)
            
            if elevation_tx > tx.el_thresh or ignore_elevation_thresh:
                k0 = tx.point_ecef(pos_vec_tx)         # we are pointing at the object when tracking
                gain_tx=tx.beam.gain(k0)     # get antenna gain
                range_tx=n.linalg.norm(pos_vec_tx)
                range_tx_p=n.linalg.norm(pos_vec_tx_p)
                range_tx_m=n.linalg.norm(pos_vec_tx_m)
                
                range_tx_dx0=n.linalg.norm(pos_vec_tx_dx0)
                range_tx_dx1=n.linalg.norm(pos_vec_tx_dx1)
                range_tx_dx2=n.linalg.norm(pos_vec_tx_dx2)                

                tx_to_target_time = range_tx/c.c
                
                # go through all receivers
                for rxi,rx in enumerate(radar._rx):
                    pos_vec_rx=-rx_p[rxi]+p
                    pos_vec_rx_p=-rx_p[rxi]+p_p
                    pos_vec_rx_m=-rx_p[rxi]+p_m

                    # rx k-vector
                    k_rec = 2.0*n.pi*pos_vec_rx/n.linalg.norm(pos_vec_rx)/tx.wavelength
                    # scattered k-vector
                    k_scat=k_rec-k_inc

                    # for linearized pos error
                    pos_vec_rx_dx0=-rx_p[rxi]+p_dx0
                    pos_vec_rx_dx1=-rx_p[rxi]+p_dx1
                    pos_vec_rx_dx2=-rx_p[rxi]+p_dx2                    
                    
                    elevation_rx=90.0-coord.angle_deg(rx_p[rxi],pos_vec_rx)
                    
                    if elevation_rx > rx.el_thresh or ignore_elevation_thresh:
                        
                        k0 = rx.point_ecef(pos_vec_rx)      # we are pointing at the object when tracking

                        gain_rx=rx.beam.gain(k0)  # get antenna gain
                        
                        range_rx=n.linalg.norm(pos_vec_rx)
                        range_rx_p=n.linalg.norm(pos_vec_rx_p)
                        range_rx_m=n.linalg.norm(pos_vec_rx_m)

                        range_rx_dx0=n.linalg.norm(pos_vec_rx_dx0)
                        range_rx_dx1=n.linalg.norm(pos_vec_rx_dx1)
                        range_rx_dx2=n.linalg.norm(pos_vec_rx_dx2)                        
                        
                        target_to_rx_time = range_rx/c.c
                        # SNR of object at measured location
                        enr_rx=debris.hard_target_enr(gain_tx,
                                                      gain_rx,
                                                      tx.wavelength,
                                                      tx.tx_power,
                                                      range_tx,
                                                      range_rx,
                                                      o.diam,
                                                      bandwidth=tx.coh_int_bandwidth,   # coherent integration bw
                                                      rx_noise_temp=rx.rx_noise)

                        if enr_rx > 1e8:
                            enr_rx=1e8

                        if enr_rx < 0.1:
                            enr_rx=0.1

                        #print("snr %1.2f"%(enr_rx))                            
                        
                        dr=10.0**(rfun(n.log10(enr_rx)))
                        ddop=10.0**(dopfun(n.log10(enr_rx)))

                        # Unknown doppler shift due to ionosphere can be up to 0.1 Hz,
                        # estimate based on typical GNU Ionospheric tomography receiver phase curves.
                        if ddop < 0.1:
                           ddop = 0.1
                        
                        dr = n.sqrt(dr**2.0 + iono_errfun(range_tx/1e3)**2.0) # add ionospheric error

                        if dr < o.diam: # if object diameter is larger than range error, make it at least as big as target
                            dr = o.diam

                        r0=range_tx+range_rx
                        rp=range_tx_p+range_rx_p
                        rm=range_tx_m+range_rx_m
                        range_rate_d=0.5*((rp-r0)+(r0-rm))/dt # symmetric numerical derivative


                        
                        # doppler (m/s) using scattering k-vector
                        range_rate = n.dot(pos_vec_rx/range_rx,state[3:6,ti]) + n.dot(pos_vec_tx/range_tx,state[3:6,ti])

                        
                        doppler = range_rate/tx.wavelength                        
                        
#                        print("rr1 %1.1f rr2 %1.1f"%(range_rate_d,range_rate))
                        doppler_k = n.dot(k_scat,ecef_vel[:,ti])/2.0/n.pi
                        range_rate_k = doppler_k*tx.wavelength

                        # for linearized errors, range rate at small perturbations to state vector velocity parameters
                        range_rate_k_dv0=tx.wavelength*n.dot(k_scat,ecef_vel[:,ti]+n.array([dv,0,0]))/2.0/n.pi
                        range_rate_k_dv1=tx.wavelength*n.dot(k_scat,ecef_vel[:,ti]+n.array([0,dv,0]))/2.0/n.pi
                        range_rate_k_dv2=tx.wavelength*n.dot(k_scat,ecef_vel[:,ti]+n.array([0,0,dv]))/2.0/n.pi
                        
                        # full range for error calculation, with small perturbations to position state
                        full_range_dx0=range_rx_dx0+range_tx_dx0
                        full_range_dx1=range_rx_dx1+range_tx_dx1
                        full_range_dx2=range_rx_dx2+range_tx_dx2
                        
                        if enr_rx > tx.enr_thresh:
                            # calculate jacobian row for state vector errors
                            # range
                            J[2*state_meas_idx,0:3]=n.array([(full_range_dx0-r0)/dx,(full_range_dx1-r0)/dx,(full_range_dx2-r0)/dx])
                            # range inverse variance
                            Sigma_Lin[2*state_meas_idx,2*state_meas_idx]=1.0/dr**2.0
                            # range-rate 
                            J[2*state_meas_idx+1,3:6]=n.array([(range_rate_k_dv0-range_rate_k)/dv,(range_rate_k_dv1-range_rate_k)/dv,(range_rate_k_dv2-range_rate_k)/dv])
                            # range-rate inverse variance
                            Sigma_Lin[2*state_meas_idx+1,2*state_meas_idx+1]=1.0/(tx.wavelength*ddop)**2.0                            
                            
                            state_meas_idx+=1

                            # detection!
                            if debug_low:
                                print("rx %d tx el %1.2f rx el %1.2f gain_tx %1.2f gain_rx %1.2f enr %1.2f rr %1.2f prop time %1.6f dr %1.2f"%(rxi,elevation_tx,elevation_rx,gain_tx,gain_rx,enr_rx,range_rate,tx_to_target_time,dr))

                            # ground foot point in geodetic
                            llh=coord.ecef2geodetic(p[0],p[1],p[2])
                            
                            # time is time of transmit pulse
                            meas[txi][rxi]["time_idx"].append(ti)
                            meas[txi][rxi]["m_time"].append(t+t0_unix)
                            meas[txi][rxi]["m_range"].append( (range_tx+range_rx)/1e3 + noise*n.random.randn()*dr/1e3 )
                            meas[txi][rxi]["m_range_std"].append( dr/1e3 )

                            rr_std=c.c*ddop/radar._tx[txi].freq/2.0/1e3 
                            meas[txi][rxi]["m_range_rate"].append(range_rate/1e3 + noise*n.random.randn()*rr_std)
                            # 0.1 m/s error due to ionosphere
                            meas[txi][rxi]["m_range_rate_std"].append(rr_std)
                            meas[txi][rxi]["m_doppler"].append(doppler + noise*n.random.randn()*ddop/1e3)
                            meas[txi][rxi]["m_doppler_std"].append(ddop)
                            meas[txi][rxi]["m_delay"].append(tx_to_target_time + target_to_rx_time)
                            meas[txi][rxi]["g_lat"].append(llh[0])
                            meas[txi][rxi]["g_lon"].append(llh[1])
                            meas[txi][rxi]["enr"].append(enr_rx)
                            
                            meas[txi][rxi]["gain_tx"].append(gain_tx)
                            meas[txi][rxi]["gain_rx"].append(gain_rx)
                            
                            true_state=n.zeros(6)
                            true_state[3:6]=(0.5*((p_p-p)+(p-p_m))/dt)/1e3
                            true_state[0:3]=p/1e3

                            meas[txi][rxi]["true_state"].append(true_state)                            
                            meas[txi][rxi]["true_time"].append(t_obs[ti]+t0_unix)
                        else:
                            if debug_high:
                                print("not detected: enr_rx {}".format(enr_rx))
                    else:
                        if debug_high:
                            print("not detected: elevation_rx {}".format(elevation_rx))
            else:
                if debug_high:
                    print("not detected: elevation_tx {}".format(elevation_tx))
                

        # if more than three measurements of range and range-rate, then we maybe able to
        # observe true state. if so, calculate the linearized covariance matrix
        if state_meas_idx > 2:
            # use only the number of measurements that were good
            JJ = J[0:(2*state_meas_idx),:]
            try:
                Sigma_post=n.linalg.inv(n.dot(n.dot(n.transpose(JJ),Sigma_Lin),JJ))
                ecef_stdevs[txi]["time_idx"].append(ti)
                ecef_stdevs[txi]["m_time"].append(t)
                ecef_stdevs[txi]["ecef_stdev"].append(Sigma_post)

            except:
                print("Singular posterior covariance...")

    if debug_low:
        print(meas)
    fnames = write_tracklets(o,meas,radar,dname,hdf5_out=hdf5_out,ccsds_out=ccsds_out)
                    
    return(meas,fnames,ecef_stdevs)

def test_envisat():
    import dpt_tools as dpt
    import space_object as so
    import radar_library as rl
    import stuffr
    mass=0.8111E+04
    diam=0.8960E+01
    m_to_A=128.651
    a=7159.5
    e=0.0001
    i=98.55
    raan=248.99
    aop=90.72
    M=47.37
    A=mass/m_to_A
    # epoch in unix seconds
    ut0=1241136000.0
    # modified julian day epoch
    
    mjd0=dpt.unix_to_jd(ut0) - 2400000.5
    print(mjd0)
    o=so.SpaceObject(a=a,e=e,i=i,raan=raan,aop=aop,mu0=M,C_D=2.3,A=A,m=mass,diam=diam,mjd0=mjd0)

    e3d = rl.eiscat_3d()
    print("EISCAT Skibotn location x,y,z ECEF (meters)")
    print(e3d._tx[0].ecef)
    ski_ecef=e3d._tx[0].ecef

    print("EISCAT Skibotn location %1.3f %1.3f %1.3f (lat,lon,alt)"%(e3d._tx[0].lat,e3d._tx[0].lon,0.0))

    t_obs=n.linspace(4440,5280,num=100)+31.974890
    t_obs2=n.linspace(4440,5280,num=100)+31.974890 + 1.0
#    t_obs=n.linspace(0,5280,num=100)
 #   t_obs2=n.linspace(0,5280,num=100)+1
    
    print("MJD %1.10f %sZ"%(mjd0+t_obs[0]/3600.0/24.0,stuffr.unix2datestr(ut0+t_obs[0])))
    ecef=o.get_state(t_obs)
    ecef2=o.get_state(t_obs2)    

    print("ECEF state x,y,z,vx,vy,vz  (km and km/s)")
    print(ecef[:,0]/1e3)    

    print("Time (UTC)          Range (km)  Vel (km/s)  ECEF X (km) ECEF Y (km) ECEF Z (km)")
    for i in range(len(t_obs)):
        dist=n.linalg.norm(ecef[0:3,i]-ski_ecef)
        dist2=n.linalg.norm(ecef2[0:3,i]-ski_ecef)

        vel=(dist2-dist)/1.0
        print("%s   %1.3f %1.3f %1.3f %1.3f %1.3f"%(stuffr.unix2datestr(ut0+t_obs[i]),dist/1e3,vel/1e3,ecef[0,i]/1e3,ecef[1,i]/1e3,ecef[2,i]/1e3))




if __name__ == "__main__":

    test_envisat()
    exit(0)
    
    import population_library as plib
    import radar_library as rlib
    import simulate_tracking

    m = plib.master_catalog()
    
    o = m.get_object(13)
    o.diam = 1.0

    e3d = rlib.eiscat_3d(beam='gauss')

    # time in seconds after mjd0
    t_all = n.linspace(0, 24*3600, num=1000)
    
    passes, _, _, _, _ = simulate_tracking.find_pass_interval(t_all, o, e3d)

    t_mid = (passes[0][0][0] + passes[0][0][1])*0.5

    t_obs = n.arange(t_mid, t_mid + 2, 0.2)

    meas, fnames, ecef_stdevs = create_tracklet(o, e3d, t_obs, hdf5_out=True, ccsds_out=True, dname="/home/danielk/IRF/E3D_PA/tmp/test")
    print(ecef_stdevs)
    print(fnames)
    print(meas)
