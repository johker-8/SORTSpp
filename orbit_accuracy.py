#!/usr/bin/env python

'''Linearized error determination for orbital elements.

These error are calculated as a function of:

 * Mean track length (track_length) seconds
 * Number of tracklets: (n_tracklets), int
 * Measurement spacing: (m_spacing), seconds

'''
import numpy as n
import matplotlib.pyplot as plt
#
# SORTS imports
#
import radar_library as rl     # radar network definition
import population_library as plib
import simulate_tracking as st # find times of possible observations
import simulate_tracklet as s  # simulate a measurement
import space_object as so      # space object


def get_inertial_basis(ecef0,ecef0_dt):
    """
    Given pos vector, and pos vector at a small positive time offset,
    calculate unit vectors for along track, normal (towards center of Earth), and cross-track directions
    """
    along_track=ecef0_dt-ecef0
    along_track=along_track/n.sqrt(along_track[0,:]**2.0+along_track[1,:]**2.0+along_track[2,:]**2.0)
    normal = ecef0/n.sqrt(ecef0[0,:]**2.0+ecef0[1,:]**2.0+ecef0[2,:]**2.0)
    cross_track=n.copy(normal)
    cross_track[:,:]=0.0
    cross_track[0,:] = along_track[1,:]*normal[2,:] - along_track[2,:]*normal[1,:]
    cross_track[1,:] = along_track[2,:]*normal[0,:] - along_track[0,:]*normal[2,:]
    cross_track[2,:] = along_track[0,:]*normal[1,:] - along_track[1,:]*normal[0,:]
    cross_track=cross_track/n.sqrt(cross_track[0,:]**2.0+cross_track[1,:]**2.0+cross_track[2,:]**2.0)
    return(along_track,normal,cross_track)

def atmospheric_errors(o,a_err_std=0.01,N_samps=200):
    """
    Estimate position errors as a function of time, assuming
    a certain error in atmospheric drag.
    """
    o0=so.SpaceObject(a=o.a,e=o.e,i=o.i,raan=o.raan,aop=o.aop,mu0=o.mu0,d=o.diam,A=o.A,m=o.m,C_D=o.C_D)
    print(o0)
    t=10**(n.linspace(2,6.2,num=100))
    t_dt=n.copy(t)+1.0     
    ecef0=o0.get_orbit(t)

    print(ecef0.shape)
    print("n_days %d"%(n.max(t)/24.0/3600.0))
    
    err=n.copy(ecef0)
    err[:,:]=0.0
    for i in range(N_samps):
        mu0=n.random.rand(1)*360.0
        print(o.m)
        print(o.C_D)
        o0=so.SpaceObject(a=o.a,e=o.e,i=o.i,raan=o.raan,aop=o.aop,mu0=mu0,d=o.diam,A=o.A,m=o.m,C_D=o.C_D)
        
        ecef0=o0.get_orbit(t)
        ecef0_dt=o0.get_orbit(t_dt)
        
        at,norm,ct=get_inertial_basis(ecef0,ecef0_dt)
        C_D=o0.C_D + o0.C_D*n.random.randn(1)[0]*a_err_std
        print(C_D)
        o1=so.SpaceObject(a=o.a,e=o.e,i=o.i,raan=o.raan,aop=o.aop,mu0=mu0,d=o.diam,A=o.A,m=o.m, C_D=C_D)
        
        ecef1=o1.get_orbit(t)
        
        err_now=(ecef1-ecef0)
        err[0,:]+=n.abs(err_now[0,:]*at[0,:]+err_now[1,:]*at[1,:]+err_now[2,:]*at[2,:])**2.0
#        err[1,:]+=n.abs(err_now[0,:]*norm[0,:]+err_now[1,:]*norm[1,:]+err_now[2,:]*norm[2,:])**2.0
        # difference in radius
        err[1,:]+=n.abs(n.sqrt(ecef0[0,:]**2.0+ecef0[1,:]**2.0+ecef0[2,:]**2.0) - n.sqrt(ecef1[0,:]**2.0+ecef1[1,:]**2.0+ecef1[2,:]**2.0))
        err[2,:]+=n.abs(err_now[0,:]*ct[0,:]+err_now[1,:]*ct[1,:]+err_now[2,:]*ct[2,:])**2.0

    ate=n.sqrt(err[0,:]/N_samps)
    if n.max(ate) > 100.0:
        idx0=n.where(ate > 100.0)[0][0]
        days=t/24.0/3600.0
    
        
    plt.loglog(t/24.0/3600.0,n.sqrt(err[0,:]/N_samps),label="Along track")
    plt.loglog(t/24.0/3600.0,n.sqrt(err[1,:]/N_samps),label="Radial")
    plt.loglog(t/24.0/3600.0,n.sqrt(err[2,:]/N_samps),label="Cross-track")
    if n.max(ate) > 100.0:    
        plt.axvline(days[idx0])
        plt.text(days[idx0],100,"$\\tau=%1.1f$ hours"%(24*days[idx0]))        
    plt.grid()
    plt.axvline(n.max(t)/24.0/3600.0)
    plt.xlim([0,n.max(t)/24.0/3600.0])
    plt.legend()
    plt.ylabel("Cartesian position error (m)")
    plt.xlabel("Time (days)")
    plt.title("a %1.0f (km) e %1.2f i %1.0f (deg) aop %1.0f (deg) raan %1.0f (deg)\nA %1.2f$\pm$ %d%% (m$^2$) mass %1.2f (kg)"%(o.a,o.e,o.i,o.aop,o.raan,o.A,int(a_err_std*100.0),o.m))
    plt.show()



    
#
# Come up with measurement times, given a specific measurement strategy
#
# n_tracklets  = number of passes to observe a target
# m_spacing    = the time interval between observations of range and range-rate
# track_length = the length of a tracklet (can at most be set_time - rise_time)
#
def create_measurements(o,radar,t0=0,track_length=1000.0, n_tracklets=2, n_meas=10, debug=False,max_time=24*3*3600.0):

    obs=st.get_passes(o,radar,t0,t0+max_time,max_dpos=100.0,debug=False)

    t=n.array(obs["t"])
    snr=n.array(obs["snr"])

    n_tx=len(radar._tx)
    n_rx=len(radar._rx)    
    n_passes=t.shape[1]
    if debug:
        print("n_passes %d"%(n_passes))

    max_track_length = n.max(t[:,:,1]-t[:,:,0])
    min_track_length = n.min(t[:,:,1]-t[:,:,0])
    if debug:
        print("maximum track length %1.2f (s)"%(max_track_length))
        print("minimum track length %1.2f (s)"%(min_track_length))    

    # what tracklets do we measure
    # each tracklet is specified with a vector of times
    tracklets=[]
    
    # these are the possible observations
    # for each transmitter
    for txi in range(n_tx):
        # for each pass
        for pi in range(n_passes):
            rise_t=t[txi,pi,0]
            set_t=t[txi,pi,1]

            max_track_length=set_t-rise_t
            this_track_length=n.min([track_length,max_track_length])
            
            # how many measurements can be made for this pass
                 
            if debug:
                print("tx %d pass %d rise %1.2f (s) set %1.2f (s) max track length %1.2f (s) track length %1.2f (s) n_meas %d"%(txi,pi,rise_t,set_t,max_track_length,this_track_length,n_meas))

            # peak snr time
            mean_obs_t=0.0

            # for each receiver
            for rxi in range(n_rx):
                peak_snr=snr[txi,pi,rxi,0]
                peak_t=snr[txi,pi,rxi,1]
                mean_obs_t+=peak_t
                if debug:
                    print("  rx %d peak_snr %1.2f (dB) peak snr time %1.2f"%(rxi,10.0*n.log10(peak_snr),peak_t))
            # the time of peak SNR
            mean_obs_t=mean_obs_t/float(n_rx)
            
            # times of observation
            m_idx=n.arange(n_meas,dtype=n.float)
            if n_meas==1:
                obs_time=n.array([mean_obs_t])
            else:
                obs_time = n.linspace(n.max([peak_t-track_length/2.0,rise_t+0.1]),n.min([peak_t+track_length/2.0,set_t-0.1]),num=n_meas)
            
            # make sure they fit the rise and set times for object
            good_idx=n.where( (obs_time > rise_t) & (obs_time < set_t))[0]
            obs_time=obs_time[good_idx]
            if len(tracklets) < n_tracklets:
                tracklets.append(obs_time)
    print("found %d tracklets"%(len(tracklets)))
    return(tracklets)

#
# Given an error covariance matric that describes the
# errors of Keplerian elements, determine the error
# covariance matrix for position in a cartesian
# coordinate system with in:
# along track, cross track, and height directions.
#
# try out different times for two hours
#
def kep_cov2cart_cov(o,Sigma_kep,t0s=n.linspace(0,2*3600,num=50)):
    # 1. Find along track, cross-track, and altitude directions at time t. We'll call this the inertial system (s)
    #
    # This coordinate system has three orthogonal unit vectors
    #
    # Assume linear transformation exists:
    #
    # s = J kep + f(kep0)
    #
    # Here f is a non-linear function of Keplerian elements, and J is the Jacobian.
    #
    # if we have errors in kep, then:
    #
    # s = J*(kep + kep_err) + f(kep0)
    # s + J*kep_err = J*kep + f(kep0)
    #
    # The errors in coordinate system s are:
    #
    # Sigma_s = J Sigma_kep J^T
    #
    # calculate a Jacobian
    Sigma_ecef=n.zeros(6)
    for t0_try in t0s:
        t0=n.array([t0_try])
        a=o.a
        e=o.e
        i=o.i
        raan=o.raan
        aop=o.aop
        mu0=o.mu0
        
        da=0.1  # 100 m
        de=1e-5
        di=1e-5
        daop=1e-5
        draan=1e-5
        dmu0=1e-5
        
        o0=so.space_object(o.a,o.e,o.i,o.raan,o.aop,o.mu0,o.diam)
        
        o_da=so.space_object(o.a+da,o.e,o.i,o.raan,o.aop,o.mu0,o.diam)
        o_de=so.space_object(o.a,o.e+de,o.i,o.raan,o.aop,o.mu0,o.diam)
        o_di=so.space_object(o.a,o.e,o.i+di,o.raan,o.aop,o.mu0,o.diam)
        o_daop=so.space_object(o.a,o.e,o.i,o.raan,o.aop+daop,o.mu0,o.diam)
        o_draan=so.space_object(o.a,o.e,o.i,o.raan+draan,o.aop,o.mu0,o.diam)
        o_dmu0=so.space_object(o.a,o.e,o.i,o.raan,o.aop,o.mu0+dmu0,o.diam)
                
        ecef0=o0.get_state(t0)[:,0]
        
        ecef_da=o_da.get_state(t0)[:,0]
        ecef_de=o_de.get_state(t0)[:,0]
        ecef_di=o_di.get_state(t0)[:,0]
        ecef_daop=o_daop.get_state(t0)[:,0]
        ecef_draan=o_draan.get_state(t0)[:,0]
        ecef_dmu0=o_dmu0.get_state(t0)[:,0]

        # Jacobian
        J=n.zeros([6,6])
        J[:,0]=(ecef_da-ecef0)/da
        J[:,1]=(ecef_de-ecef0)/de
        J[:,2]=(ecef_di-ecef0)/di
        J[:,3]=(ecef_daop-ecef0)/daop
        J[:,4]=(ecef_draan-ecef0)/draan
        J[:,5]=(ecef_dmu0-ecef0)/dmu0

        Sigma_ecef_try=n.dot(n.dot(J,Sigma_kep),n.transpose(J))
        S_diag=n.diag(Sigma_ecef_try)
        # selected largest
        Sigma_ecef=n.maximum(Sigma_ecef,S_diag)
    return(Sigma_ecef)

#
# Estimate linearized orbital elements errors, given measurements of object
# use state vector
#
def linearized_errors(o,radar,tracklets,plot=True,debug=False,t0s=n.linspace(0,2*3600,num=50),time_vector=False):

    n_tx=len(radar._tx)
    n_rx=len(radar._rx)

    # calculate a Jacobian
    a=o.a
    e=o.e
    i=o.i
    raan=o.raan
    aop=o.aop
    mu0=o.mu0
    
    da=1e-3       # 1 m
    de=1e-6
    di=1e-6
    daop=1e-6
    draan=1e-6    
    dmu0=1e-6

    o0=so.space_object(o.a,o.e,o.i,o.raan,o.aop,o.mu0,o.diam)
    o_da=so.space_object(o.a+da,o.e,o.i,o.raan,o.aop,o.mu0,o.diam)
    o_de=so.space_object(o.a,o.e+de,o.i,o.raan,o.aop,o.mu0,o.diam)
    o_di=so.space_object(o.a,o.e,o.i+di,o.raan,o.aop,o.mu0,o.diam)
    o_daop=so.space_object(o.a,o.e,o.i,o.raan,o.aop+daop,o.mu0,o.diam)
    o_draan=so.space_object(o.a,o.e,o.i,o.raan+draan,o.aop,o.mu0,o.diam)
    o_dmu0=so.space_object(o.a,o.e,o.i,o.raan,o.aop,o.mu0+dmu0,o.diam)

    # calculate the number of rows in the theory matrix
    n_rows=0
    for ti,t_obs in enumerate(tracklets):
        if debug:
            print("tracklet %d"%(ti))
            print(t_obs)

        meas0,fnames,e=s.create_tracklet(o0,radar,t_obs,dt=0.01,hdf5_out=False,ccsds_out=False,dname="./tracklets",noise=False)

        for txi in range(n_tx):
            for rxi in range(n_rx):
                n_meas=len(meas0[txi][rxi]["m_time"])
                if debug:
                    print("n_meas %d"%(n_meas))
                n_rows+=n_meas

    # one range and one range rate measurement
    n_meas=n_rows*2
    if debug:
        print("number of measurements %d"%(n_meas))
    # error covariance matrix of measurements
    Sigma_m_inv = n.zeros([n_meas,n_meas])
    # jacobian
    J = n.zeros([n_meas,6])

    row_idx=0
    # perturb elements in each of the six parameters
    for ti,t_obs in enumerate(tracklets):
        if debug:
            print("Tracklet %d"%(ti))
        meas0,fnames,e=s.create_tracklet(o0,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False)       
        meas_da,fnames,e=s.create_tracklet(o_da,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        meas_de,fnames,e=s.create_tracklet(o_de,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        meas_di,fnames,e=s.create_tracklet(o_di,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        meas_daop,fnames,e=s.create_tracklet(o_daop,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        meas_draan,fnames,e=s.create_tracklet(o_draan,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        meas_dmu0,fnames,e=s.create_tracklet(o_dmu0,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        
        for txi in range(n_tx):
            for rxi in range(n_rx):
                n_meas=len(meas0[txi][rxi]["m_time"])
                if debug:
                    print("n_meas %d measurement %d"%(n_meas,row_idx))
                for mi in range(n_meas):
                    # range and range-rate error
                    range_std=meas0[txi][rxi]["m_range_std"][mi]
                    range_rate_std=meas0[txi][rxi]["m_range_rate_std"][mi]

                    # range and range rate error variance
                    Sigma_m_inv[2*row_idx,2*row_idx]=1.0/range_std**2.0
                    Sigma_m_inv[2*row_idx+1,2*row_idx+1]=1.0/range_rate_std**2.0

                    # apogee
                    m_range_da=(meas_da[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/da
                    m_range_rate_da=(meas_da[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/da
                    # e
                    m_range_de=(meas_de[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/de
                    m_range_rate_de=(meas_de[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/de
                    # inc
                    m_range_di=(meas_di[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/di
                    m_range_rate_di=(meas_di[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/di
                    # aop
                    m_range_daop=(meas_daop[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/daop
                    m_range_rate_daop=(meas_daop[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/daop
                    # raan
                    m_range_draan=(meas_draan[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/draan
                    m_range_rate_draan=(meas_draan[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/draan
                    # mu0
                    m_range_dmu0=(meas_dmu0[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/dmu0
                    m_range_rate_dmu0=(meas_dmu0[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/dmu0
                    if debug:
                        print("r da %1.2f di %1.2f de %1.2f daop %1.2f draan %1.2f dmu0 %1.2f"%(m_range_da,m_range_di,m_range_de,m_range_daop,m_range_draan,m_range_dmu0))
                        print("rr da %1.2f di %1.2f de %1.2f daop %1.2f draan %1.2f dmu0 %1.2f"%(m_range_rate_da,m_range_rate_di,m_range_rate_de,m_range_rate_daop,m_range_rate_draan,m_range_rate_dmu0))
                    
                    J[2*row_idx,0]=m_range_da
                    J[2*row_idx,1]=m_range_de
                    J[2*row_idx,2]=m_range_di
                    J[2*row_idx,3]=m_range_daop
                    J[2*row_idx,4]=m_range_draan
                    J[2*row_idx,5]=m_range_dmu0
                    
                    J[2*row_idx+1,0]=m_range_rate_da
                    J[2*row_idx+1,1]=m_range_rate_de
                    J[2*row_idx+1,2]=m_range_rate_di
                    J[2*row_idx+1,3]=m_range_rate_daop
                    J[2*row_idx+1,4]=m_range_rate_draan
                    J[2*row_idx+1,5]=m_range_rate_dmu0                 
                    row_idx+=1

    # linearized error covariance
    Sigma_pos=n.linalg.inv(n.dot(n.dot(n.transpose(J),Sigma_m_inv),J))

    # linear transformation
    Sigma_ecef = kep_cov2cart_cov(o,Sigma_pos,t0s=t0s)
    
    return(Sigma_pos, Sigma_ecef)

# plot measurements
def plot_measurements(o,r,tracklets):
    n_tx=len(r._tx)
    n_rx=len(r._rx)

    # plot range meas
    plt.subplot(231)
    plt.ylabel("Range (km)")
    plt.xlabel("Time (s)")        
    for ti,t_obs in enumerate(tracklets):
        print("Plotting tracklet %d"%(ti))
        meas,fnames,e=s.create_tracklet(o,r,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        
        for txi in range(n_tx):
            for rxi in range(n_rx):
                plt.plot(meas[txi][rxi]["m_time"],meas[txi][rxi]["m_range"],".")

    plt.subplot(232)
    plt.ylabel("Range-rate (km/s)")
    plt.xlabel("Time (s)")            
    for ti,t_obs in enumerate(tracklets):
        print("Tracklet %d"%(ti))
        meas,fnames,e=s.create_tracklet(o,r,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        
        for txi in range(n_tx):
            for rxi in range(n_rx):
                plt.plot(meas[txi][rxi]["m_time"],meas[txi][rxi]["m_range_rate"],".")

    plt.subplot(233)
    plt.xlabel("Ground longitude (deg)")
    plt.ylabel("Ground latitude (deg)")    
    for ti,t_obs in enumerate(tracklets):
        meas,fnames,e=s.create_tracklet(o,r,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        for txi in range(n_tx):
            for rxi in range(n_rx):
                plt.plot(meas[txi][rxi]["g_lon"],meas[txi][rxi]["g_lat"],".")
                
                
    plt.subplot(234)
    plt.ylabel("Range error (m)")
    plt.xlabel("Time (s)")            
    for ti,t_obs in enumerate(tracklets):
        meas,fnames,e=s.create_tracklet(o,r,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        
        for txi in range(n_tx):
            for rxi in range(n_rx):
                plt.plot(meas[txi][rxi]["m_time"],n.array(meas[txi][rxi]["m_range_std"])*1e3,".")
    plt.subplot(235)
    plt.ylabel("Range-rate error (m/s)")
    plt.xlabel("Time (s)")            
    for ti,t_obs in enumerate(tracklets):
        meas,fnames,e=s.create_tracklet(o,r,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        
        for txi in range(n_tx):
            for rxi in range(n_rx):
                plt.plot(meas[txi][rxi]["m_time"],n.array(meas[txi][rxi]["m_range_rate_std"])*1e3,".")
                
    plt.subplot(236)
    plt.ylabel("SNR (dB)")
    plt.xlabel("Time (s)")
    for ti,t_obs in enumerate(tracklets):
        meas,fnames,e=s.create_tracklet(o,r,t_obs,hdf5_out=False,ccsds_out=False,noise=False)
        
        for txi in range(n_tx):
            for rxi in range(n_rx):
                plt.plot(meas[txi][rxi]["m_time"],10.0*n.log10(n.array(meas[txi][rxi]["enr"])),".")
                
    plt.tight_layout()
    # tdb, add covariance ellipse
    plt.show()
    

#
# Optimal observation strategy, given a
# fixed number of measurements: n_meas
#
# What is the optimal track length and number of tracks.
# The answer: the more tracks, the better. Shorter track length is also usually better,
# because the measurements are concentrated in the high SNR region of the pass.
#
def error_sweep(o,r,n_meas=10):

    track_lengths=[1,10,50,100,200,400,600,1000]
    n_meas = 10
    n_tracklets=[1,2,5,10]
    S=n.zeros([len(n_tracklets),len(track_lengths)])

    n_total=0
    for ti,n_tracks in enumerate(n_tracklets):
        n_per_tracklet=n.max([int(n.round(n_meas/n_tracks)),1])
        for tli,track_len in enumerate(track_lengths):
            # create measurements that match specifications
            tracklets=create_measurements(o,r, track_length=track_len, n_tracklets=n_tracks, n_meas=n_per_tracklet)
            print(len(tracklets))
            n_actual_meas=0
            for mti in range(n_tracks):
                n_actual_meas+=len(tracklets[mti])
            print("n_meas %d n_tracklets %d n_per_tracklet %d track_len %f"%(n_per_tracklet*n_tracks,n_tracks,n_per_tracklet,track_len))                
            print("n_tracklets %d number of actual meas %d"%(len(tracklets),n_actual_meas))
            Sigma_kep,Sigma_ecef=linearized_errors(o,r,tracklets)

            max_std=n.sqrt(n.max(Sigma_ecef))
            S[ti,tli]=max_std


    plt.pcolormesh(n.concatenate([[0],track_lengths]),n.concatenate([[0],n_tracklets]),n.log10(S))
    plt.title("ECEF position error std dev (log10)")
    plt.xlabel("Track length (s)")
    plt.ylabel("Number of passes")    
    plt.colorbar()
    plt.show()


#
# Determine how variance differs as a function of n_meas
#
# Fixed track length (100 sec)
#
def error_sweep_n(o,r):

    track_length=200.0
    n_tracklets=[1,2,5,10]
    n_meass=[10,20,40,80,160,320]
    S=n.zeros([len(n_tracklets),len(n_meass)])

    t0s=[0]
    for t0 in t0s:
        for ti,n_tracks in enumerate(n_tracklets):
            for tli,n_meas in enumerate(n_meass):
                n_per_tracklet=n.max([int(n.round(n_meas/n_tracks)),1])
                # create measurements that match specifications
                tracklets=create_measurements(o,r, t0=t0, track_length=track_length, n_tracklets=n_tracks, n_meas=n_per_tracklet)
                #            print(len(tracklets))
                n_actual_meas=0
                for mti in range(n_tracks):
                    n_actual_meas+=len(tracklets[mti])
                print("n_meas %d n_tracklets %d n_per_tracklet %d track_len %f"%(n_per_tracklet*n_tracks,n_tracks,n_per_tracklet,track_length))                
                print("n_tracklets %d number of actual meas %d"%(len(tracklets),n_actual_meas))
                Sigma_kep,Sigma_ecef=linearized_errors(o,r,tracklets)
                max_std=n.sqrt(n.max(Sigma_ecef))
                
                print("max std %1.2f"%(max_std))
                
                # give sqrt of maximum eigenvalue of covariance matrix
                S[ti,tli]+=n.sqrt(max_std)
    # average
    S=S/float(len(t0s))


    plt.pcolormesh(n.concatenate([[0],n_meass]),n.concatenate([[0],n_tracklets]),n.log10(S))
    plt.title("ECEF position error std dev (log10)")
    plt.xlabel("Number of measurements")
    plt.ylabel("Number of passes")    
    plt.colorbar()
    plt.show()



#
# Determine how variance differs as a function of n_meas
#
# Fixed track length (100 sec)
#
def error_sweep_track_length(o,r):
    track_lengths=n.linspace(1,1000,num=25)
    n_tracklets=[4,5,6,8]
    n_meas=10
    
    S=n.zeros([len(track_lengths),len(n_tracklets)])

    t0s=[0]
    for t0 in t0s:
        for ti,track_length in enumerate(track_lengths):
            for tli,n_tracks in enumerate(n_tracklets):
                tracklets=create_measurements(o,r, t0=t0, track_length=track_length, n_tracklets=n_tracks, n_meas=n_meas)
                Sigma_kep,Sigma_ecef=linearized_errors(o,r,tracklets)
                max_std=n.sqrt(n.max(Sigma_ecef))
                print("max std %1.2f"%(max_std))
                S[ti,tli]=max_std
            
    # average
    S=S/float(len(t0s))
    for i in range(len(n_tracklets)):
        plt.semilogy(track_lengths,S[:,i],label="%d tracklets"%(n_tracklets[i]))
    plt.legend()
    plt.ylim([5,200])
    plt.title("Error as a function of track length ($N_{\mathrm{m}}=%d$)"%(n_meas))
    plt.xlabel("Maxmimum track length $T$ (s)")
    plt.ylabel("Worst-case position error (meters)")
    plt.show()

#
# Determine how variance differs as a function of n_meas
#
# Fixed track length (100 sec)
#
def error_sweep_n_meas(o,r):
    track_length=1000.0
    n_tracklets=[1,3,4,5,6,8]
    n_meas=[2,3,4,5,6,7,8,16,32,64,128,256]
    
    S=n.zeros([len(n_meas),len(n_tracklets)])

    t0s=[0]
    for t0 in t0s:
        for tli,n_tracks in enumerate(n_tracklets):
            for mi,nm in enumerate(n_meas):            
                tracklets=create_measurements(o,r, t0=t0, track_length=track_length, n_tracklets=n_tracks, n_meas=nm)
                Sigma_kep,Sigma_ecef=linearized_errors(o,r,tracklets)
                max_std=n.sqrt(n.max(Sigma_ecef))
                print("max std %1.2f"%(max_std))
                S[mi,tli]=max_std
            
    # average
    S=S/float(len(t0s))
    for i in range(len(n_tracklets)):
        plt.loglog(n_meas,S[:,i],label="%d tracklets"%(n_tracklets[i]))
    plt.legend()
    plt.ylim([5,10e3])
    plt.title("Error as a function of $N_{\mathrm{m}}$")
    plt.xlabel("Number of measurements per tracklet $N_{\mathrm{m}}$")
    plt.ylabel("Worst-case position error (meters)")
    plt.show()


#
# Determine how variance differs as a function of n_meas
#
# Fixed track length (100 sec)
#
def error_sweep_n_meas_constn(o,r):
    
    track_length=1000.0
    n_tracklets=[1,4,8]
    n_meas=[8,16,32,64,128,256]
    
    S=n.zeros([len(n_meas),len(n_tracklets)])

    t0s=[0]
    
    for t0 in t0s:
        for mi,nm in enumerate(n_meas):
            for tli,n_tracks in enumerate(n_tracklets):
                n_per=nm/float(n_tracks)
                if n_per < 1.0:
                    S[mi,tli]=n.nan
                else:
                    tracklets=create_measurements(o,r, t0=t0, track_length=track_length, n_tracklets=n_tracks, n_meas=int(n_per))
                    Sigma_kep,Sigma_ecef=linearized_errors(o,r,tracklets)
                    max_std=n.sqrt(n.max(Sigma_ecef))
                    S[mi,tli]=max_std

            
    # average
    S=S/float(len(t0s))
    for i in range(len(n_tracklets)):
        plt.loglog(n_meas,S[:,i],label="%d tracklets"%(n_tracklets[i]))
    plt.legend()
    plt.ylim([5,10e3])
    plt.title("Error as a function of $N_{\mathrm{m}}N_{t}$")
    plt.ylabel("Worst-case position error (meters)")
    plt.xlabel("Total number of measurements $N_{\mathrm{m}} N_{t}$")
    plt.show()



def error_sweep_time(o,r):
    
    track_length=1000.0
    n_tracklets=100
    n_meas=10

    t0s=n.linspace(0,14*24*3600,num=100)    
    S=n.zeros(len(t0s))

#    create_measurements(o,r,t0=0,track_length=1000.0, n_tracklets=2, n_meas=10, debug=False,max_time=24*3*3600.0):
    
    tracklets=create_measurements(o,r, t0=0, track_length=track_length, n_tracklets=n_tracklets, n_meas=n_meas,max_time=24*7*3600.0)

#    tracklets=[tracklets[5],tracklets[11],tracklets[22],tracklets[33],tracklets[44]]
    tracklets=[tracklets[30],tracklets[31],tracklets[32],tracklets[33],tracklets[34]]    
    
    print(tracklets)
    for ti,t0 in enumerate(t0s):
        Sigma_kep,Sigma_ecef=linearized_errors(o,r,tracklets,t0s=[t0],time_vector=True)
        max_std=n.sqrt(n.max(Sigma_ecef))
        print("t0 %1.2f s %1.2f"%(t0/24.0/3600.0,max_std))
        S[ti]=max_std
    for t in tracklets:
        plt.axvline(n.mean(t)/3600.0/24.0)
    plt.plot(t0s/3600.0/24.0,S)
    plt.ylabel("Worst-case position error (meters)")
    plt.xlabel("Time (days)")    
    plt.show()
    
    

    

# TBD, calculate errors for initial position after detection with one single point measurement
# errors in ecef pos and ecef vel
    
if __name__ == "__main__":
    
    # space object population
    o=so.SpaceObject(a=6670.0, e=1e-4, i=89.0, raan=12.0, aop=23.0, mu0=32.0, A=10**(-2.0), m=1.0, d=1.0)        
    atmospheric_errors(o,a_err_std=0.05)
    exit(0)
    
    # radar network
    r=rl.eiscat_3d()
    error_sweep_time(o,r, a_err_std=0.05)    
#    error_sweep_n_meas_constn(o,r)
 #   error_sweep_n_meas(o,r)
#    error_sweep_track_length(o,r)
 #   error_sweep_n(o,r)
    

    #error_sweep_n(o,r)
    #error_sweep(o,r)
    # create measurements that match specifications
    tracklets=create_measurements(o,r, track_length=1000.0, n_tracklets=1, n_meas=100)
    print(tracklets)

    plot_measurements(o,r,tracklets)

    # error covariance matrix in keplerian and ecef coordinates
    Sigma_kep,Sigma_ecef=linearized_errors(o,r,tracklets)
    print(Sigma_ecef)
    print("Keplerian elements stdev")
    print(n.sqrt(n.diag(Sigma_kep)))
    print("Cartesian ECEF state stdev")    
    print(n.sqrt(Sigma_ecef))
