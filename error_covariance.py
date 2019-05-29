#!/usr/bin/env python

import numpy as n
import space_object as so      # space object

import radar_library as rl     # radar network definition
import population_library as plib
import simulate_tracking as st # find times of possible observations
import simulate_tracklet as stra
import matplotlib.pyplot as plt
import dpt_tools as dpt
import os
import glob
import h5py
import atmospheric_drag_error_model as ad
import atmospheric_drag_errors as ade
import scipy.optimize as sio
from mpi4py import MPI

import propagator_sgp4 as ps

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def get_t_obs(o, radar, n_tracklets=1, track_length=600.0, n_points=3, debug=False, h0=0.0,h1=24.0, half_track=False, sort_by_length=False, n_points0=50):
    """
    Weighted linear least squares estimation of orbital elements
    
    Simulate measurements using create tracklet and estimate 
    orbital parameters, which include six keplerian and area to mass ratio. 

    Use fmin search. 
    Optionally utilize MCMC to sample the distribution of parameters.
    
    number of tracklets, tracklet length, and number of tracklet points per tracklet are
    user definable, allowing one to try out different measurement strategies. 
    """
    e3d = rl.eiscat_3d(beam='gauss')

    # time in seconds after mjd0
    t_all = n.linspace(h0*3600, h1*3600.0, num=1000*(h1-h0))
    
    passes, _, _, _, _ = st.find_pass_interval(t_all, o, e3d)
    if debug:
        print("number of possible tracklets %d"%(len(passes[0])))

    if n_tracklets == None:
        n_tracklets = len(passes[0])

    tracklet_idx = n.arange(len(passes[0]))
    
    n_tracklets = n.min([n_tracklets,len(passes[0])])

    tracklet_lengths=[]
    for pi in tracklet_idx:
        p = passes[0][pi]
        tracklet_lengths.append(p[1]-p[0])
    tracklet_lengths=n.array(tracklet_lengths)

    if sort_by_length:
        idx=n.argsort(tracklet_lengths)[::-1]
    else:
        idx=n.arange(len(tracklet_lengths))
    # select n_tracklets longest tracklets
    tracklet_idx=tracklet_idx[idx[0:n_tracklets]]
    
    t_obss=[]
    t_means=[]
    for pii,pi in enumerate(tracklet_idx):
        p = passes[0][pi]
        mean_t=0.5*(p[1]+p[0])
        t_means.append(mean_t)
        if debug:
            print("duration %1.2f"%(p[1]-p[0]))
        if p[1]-p[0] > 5.0:
            if n_points == 1:
                t_obs=n.array([mean_t])
            elif pii == 0 and half_track:
                # maximize track length, but only observe half of the pass (simulate initial discovery follow-up measurments)
                t_obs=n.linspace(mean_t, n.min([p[1],mean_t+track_length/2]),num=n_points0)
            else:
                # maximize track length
                t_obs=n.linspace(n.max([p[0],mean_t-track_length/2]), n.min([p[1],mean_t+track_length/2]),num=n_points)
            t_obss=n.concatenate([t_obss,t_obs])
    return(t_obss,n_tracklets,n.array(t_means),tracklet_lengths[0:n_tracklets])


def change_of_epoch_test(o_in,t0,plot=False):
    """
    return a new space object with cartesian state of o at t0 at t0=0
           use minimization procedure to ensure that the conversion results 
           in minimal state vector error along a 24 hour orbit.
    """
    o=o_in.copy()

    tfit=n.linspace(0,24*3600,num=100)
    ecef0=o.get_state(t0+tfit)/1e3
    o.propagate(t0,frame_transformation="TEME")
    

    def ss(x):
        try:
            o.update(x=x[0], y=x[1], z=x[2], vx=x[3], vy=x[4], vz=x[5])
            ecefc=o.get_state(tfit)/1e3
            s=n.sum(n.abs(ecef0[0:3,:]-ecefc[0:3,:])**2.0)+1e2*n.sum(n.abs(ecef0[3:6,:]-ecefc[3:6,:])**2.0)
            return(s)
        except:
            return(1e99)

    xhat=sio.fmin(ss,[o.x, o.y, o.z, o.vx, o.vy, o.vz],ftol=1e-11,xtol=1e-11,disp=False)
    ssq=ss(xhat)
    print("oid %d sum of squares: %f"%(o.oid,ssq))
    
    ecef1=o.get_state(tfit)/1e3

    
    if plot:
        plt.subplot(121)
        for i in range(3):
            plt.plot(ecef0[i,:]-ecef1[i,:])
            plt.ylabel("Position error (km)")
        plt.subplot(122)
        for i in range(3,6):
            plt.plot(ecef0[i,:]-ecef1[i,:])
            plt.ylabel("Position error (km/s)")            
        plt.show()
        
    return(o)


def linearized_errors(ot,radar,t_obs,plot=True,debug=False,t0s=n.linspace(0,2*3600,num=50),
                      debug_output=False,
                      change_epoch=False,
                      log10_A_to_m_std=0.59,  # estimated from detectable population in master 2009
                      time_vector=False, dx=0.0001, dv=1e-6, dcd=0.01, include_drag=False):
    """ 
    Use numerical differences to estimate the measurement error covariance matrix using
    cartesian state vector at epoch plus area-to-mass ratio (7 orbital parameters)

    Use a prior for area-to-mass ratio, to allow also estimating covariance with one tri-static measurement.
    """
    n_tx=len(radar._tx)
    n_rx=len(radar._rx)

    o=ot.copy()
    
    Am = o.A/o.m
    o.m=1.0
    o.A=Am
    
    o0=o.copy()

    o_dx0=o.copy()
    o_dx0.update(x=o.x+dx)
    
    o_dx1=o.copy()
    o_dx1.update(y=o.y+dx)
    
    o_dx2=o.copy()
    o_dx2.update(z=o.z+dx)
    
    o_dv0=o.copy()
    o_dv0.update(vx=o.vx+dv)
    
    o_dv1=o.copy()
    o_dv1.update(vy=o.vy+dv)
    
    o_dv2=o.copy()
    o_dv2.update(vz=o.vz+dv)

    # area to mass ratio
    o_dcd=o.copy()

    AmL=n.log10(Am)
    AmL2=AmL + dcd
    o_dcd.A = 10**(AmL2)

    # create measurements
    meas0,fnames,e=stra.create_tracklet(o0,radar,t_obs,dt=0.01,hdf5_out=False,ccsds_out=False,dname="./tracklets",noise=False, ignore_elevation_thresh=True)

    n_rows=0
    # figure out how many measurements we have
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

    # create jacobian
    if include_drag:
        # include prior for A/m, and one extra parameter for log10(A/m)
        J = n.zeros([n_meas+1,7])
        Sigma_m_inv = n.zeros([n_meas+1,n_meas+1])
    else:
        # only a cartesian state
        J = n.zeros([n_meas,6])
        Sigma_m_inv = n.zeros([n_meas,n_meas])        

    # perturb elements in each of the six parameters
    meas_dx0,fnames,e=stra.create_tracklet(o_dx0,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False, ignore_elevation_thresh=True)
    meas_dx1,fnames,e=stra.create_tracklet(o_dx1,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False, ignore_elevation_thresh=True)
    meas_dx2,fnames,e=stra.create_tracklet(o_dx2,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False, ignore_elevation_thresh=True)
    meas_dv0,fnames,e=stra.create_tracklet(o_dv0,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False, ignore_elevation_thresh=True)
    meas_dv1,fnames,e=stra.create_tracklet(o_dv1,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False, ignore_elevation_thresh=True)
    meas_dv2,fnames,e=stra.create_tracklet(o_dv2,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False, ignore_elevation_thresh=True)
    meas_dcd,fnames,e=stra.create_tracklet(o_dcd,radar,t_obs,hdf5_out=False,ccsds_out=False,noise=False, ignore_elevation_thresh=True)

    # Populate Jacobian
    # m = Jx + f(x_map) + \xi
    row_idx=0
    range_0=0.0
    for txi in range(n_tx):
        for rxi in range(n_rx):
            n_meas0=len(meas0[txi][rxi]["m_time"])
            # save initial range
            if txi==0 and txi==0:
                if debug:
                    print("range std %f (m) range-rate std %f (m/s)"%(meas0[txi][rxi]["m_range_std"][0]*1e3,meas0[txi][rxi]["m_range_rate_std"][0]*1e3))
                range_0=meas0[txi][rxi]["m_range"][0]
            
            n_meas1=len(meas_dx0[txi][rxi]["m_time"])
            n_meas2=len(meas_dx1[txi][rxi]["m_time"])
            n_meas3=len(meas_dx2[txi][rxi]["m_time"])
            n_meas4=len(meas_dv0[txi][rxi]["m_time"])
            n_meas5=len(meas_dv1[txi][rxi]["m_time"])
            n_meas6=len(meas_dv2[txi][rxi]["m_time"])
            n_meas7=len(meas_dcd[txi][rxi]["m_time"])

            if debug:
                print("n_meas %d measurement %d"%(n_meas0,row_idx))
                
            for mi in range(n_meas0):
                # range and range-rate error
                range_std=meas0[txi][rxi]["m_range_std"][mi]
                range_rate_std=meas0[txi][rxi]["m_range_rate_std"][mi]

#                print("range_std %1.1f range_rate_std %1.1f"%(range_std*1e3,range_rate_std*1e3))
 #               print("range %1.1f range_rate %1.1f"%(meas0[txi][rxi]["m_range"][mi],meas0[txi][rxi]["m_range_rate"][mi]))
                
                # range and range rate error variance
                Sigma_m_inv[2*row_idx,2*row_idx]=1.0/range_std**2.0
                Sigma_m_inv[2*row_idx+1,2*row_idx+1]=1.0/range_rate_std**2.0

                # siple first order difference derivate estimate
                m_range_dx0=(meas_dx0[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/dx
                m_range_rate_dx0=(meas_dx0[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/dx
                
                m_range_dx1=(meas_dx1[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/dx
                m_range_rate_dx1=(meas_dx1[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/dx
                
                m_range_dx2=(meas_dx2[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/dx
                m_range_rate_dx2=(meas_dx2[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/dx
                
                m_range_dv0=(meas_dv0[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/dv
                m_range_rate_dv0=(meas_dv0[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/dv
                
                m_range_dv1=(meas_dv1[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/dv
                m_range_rate_dv1=(meas_dv1[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/dv

                m_range_dv2=(meas_dv2[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/dv
                m_range_rate_dv2=(meas_dv2[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/dv

                m_range_dcd=(meas_dcd[txi][rxi]["m_range"][mi] - meas0[txi][rxi]["m_range"][mi])/dcd
                m_range_rate_dcd=(meas_dcd[txi][rxi]["m_range_rate"][mi] - meas0[txi][rxi]["m_range_rate"][mi])/dcd

                J[2*row_idx,0]=m_range_dx0
                J[2*row_idx,1]=m_range_dx1
                J[2*row_idx,2]=m_range_dx2
                J[2*row_idx,3]=m_range_dv0
                J[2*row_idx,4]=m_range_dv1
                J[2*row_idx,5]=m_range_dv2
                if include_drag:
                    J[2*row_idx,6]=m_range_dcd
                
                J[2*row_idx+1,0]=m_range_rate_dx0
                J[2*row_idx+1,1]=m_range_rate_dx1
                J[2*row_idx+1,2]=m_range_rate_dx2
                J[2*row_idx+1,3]=m_range_rate_dv0
                J[2*row_idx+1,4]=m_range_rate_dv1
                J[2*row_idx+1,5]=m_range_rate_dv2
                if include_drag:                
                    J[2*row_idx+1,6]=m_range_rate_dcd
                row_idx+=1
    #
    # Prior information about log10(A/m) parameter
    # This is achieved by adding a virtual measurement:
    # - we've measured log10(A/m) with 0.77 standard deviation
    # 
    # Bayesian statistics, but with linear regression.
    # 
    if include_drag:
        J[n_meas,6]=1.0
        
        Sigma_m_inv[n_meas,n_meas]=1.0/(log10_A_to_m_std**2.0)


    # linearized error covariance
    # \Sigma_post = (A^T \Sigma^{-1} A)^{-1} 
    Sigma_pos=n.linalg.inv(n.dot(n.dot(n.transpose(J),Sigma_m_inv),J))

    if debug_output:
        print("ECEF error stdev")
        print(n.sqrt(n.diag(Sigma_pos))[0:6]*1e3)
        if include_drag:
            print("Drag coefficient error")    
            print(n.sqrt(n.diag(Sigma_pos))[6])
    
    return(Sigma_pos,range_0)


def sample_orbit(o,C,t_means,N=100,alpha=4.1,beta=7.8,t1=8943.65,
                 plot=False,
                 plot_t_means=[],
                 max_error=None,
                 t_obs=n.linspace(-8*24*3600,8*24*3600,num=1000),
                 use_atmospheric_errors=False,
                 fname=None,
                 save_if_okay=False,
                 mean_error_threshold=100.0):

    """
    Propagate error covariance in time.
    Sample mean position error.
    Optionally add atmospheric drag uncertainty related errors.
    """

    # where to examine errors
    t_idx=n.where( (t_obs > 0) & (t_obs < 24*3600) )[0]
    
    ecef0=o.get_state(t_obs)
    Am=o.A/o.m
    AmL=n.log10(Am)
    o.m=1.0
    o.A=Am
    o1=o.copy()
    diff2=n.zeros(len(t_obs))
    for i in range(N):
        p=n.random.multivariate_normal(n.zeros(C.shape[0]),C)
        o1.update(x=o.x+p[0],
                  y=o.y+p[1],
                  z=o.z+p[2],
                  vx=o.vx+p[3],
                  vy=o.vy+p[4],
                  vz=o.vz+p[5])
        # include drag error, if present
        if C.shape[0] == 7:
            o1.A=10**(AmL+p[6])
            
        ecef1=o1.get_state(t_obs)
        diff2+=(ecef1[0,:]-ecef0[0,:])**2.0 +  (ecef1[0,:]-ecef0[0,:])**2.0 + (ecef1[0,:]-ecef0[0,:])**2.0

    if use_atmospheric_errors:
        atmos_err_stdev=ad.atmospheric_errors_rw(t_means, t=t_obs, alpha=o.alpha, beta=o.beta, t1=o.t1, error0=1.0,loglog=False, show_model=False,plot=False)
        atmos_err_stdev[n.where(n.isnan(atmos_err_stdev))[0]]=0.0
        total_err_stdev=n.sqrt(atmos_err_stdev**2.0 + diff2/N)
    else:
        total_err_stdev=n.sqrt(diff2/N)
    mean_error=n.mean(total_err_stdev[t_idx])
    maxim_error=n.max(total_err_stdev[t_idx])
    last_error=total_err_stdev[-1]

    do_plot=True
    if save_if_okay:
        do_plot=False
        if (mean_error < mean_error_threshold):
            do_plot=True
            
    if plot and do_plot:
        plt.figure(figsize=(10,5))
#        for tmi,t in enumerate(t_means):
 #           plt.axvline(t/3600.0,color="C3")
        for tmi,t in enumerate(plot_t_means):
            plt.axvline(t/3600.0,color="C3")
        

        if use_atmospheric_errors:
            plt.semilogy(t_obs/3600,atmos_err_stdev,label="atmos error",color="C1")

            plt.semilogy(t_obs/3600,n.sqrt(diff2/N),label="orbit error",color="C0")
            plt.semilogy(t_obs/3600,total_err_stdev,label="total error",color="C2")            
        else:
            plt.semilogy(t_obs/3600,n.sqrt(diff2/N),label="total error",color="C2")            
        if max_error != None:
            plt.axhline(max_error,color="black")
        plt.legend()
        plt.title("oid=%d a=%1.0f (km) log$_{10}$(e)=%1.1f i=%1.0f (deg) d=%1.2f (m) log$_{10}$(A/m)=%1.0f (m$^2$/kg)\n mean error %1.0f (m) "%(o.oid,o.a,n.log10(o.e),o.i,o.d,n.log10(o.A/o.m),mean_error))
        plt.xlabel("Time (h)")
        plt.ylabel("Position error standard deviation (m)")
        plt.tight_layout()
        if fname == None:
            plt.show()
        else:
            plt.savefig(fname)
            plt.clf()
            plt.close()
    return(mean_error,last_error,maxim_error)


def error_sweeps(sweep_points=[1,3,10,50,100]):
    """
    Answer the question: what measurement strategy is the best?
    """
    h=h5py.File("master/drag_info.h5","r")
    radar_e3d = rl.eiscat_3d(beam='interp', stage=1)

    # space object population
    pop_e3d = plib.filtered_master_catalog_factor(
        radar = radar_e3d,   
        treshhold = 0.01,    # min diam
        min_inc = 50,
        prop_time = 48.0,
    )
    
    t_sample=n.linspace(0,24*3600.0,num=1000)
    for o in pop_e3d.object_generator():

        # get drag info
        idx=n.where(h["oid"].value == o.oid)[0]
        o.alpha=h["alpha"].value[idx]
        o.t1=h["t1"].value[idx]
        o.beta=h["beta"].value[idx]                
        
        for spi,sp in enumerate(sweep_points):
            print(spi)
            all_t_obs,all_n_tracklets,all_t_means,all_t_lens=get_t_obs(o, radar_e3d, n_tracklets=50, track_length=3600.0, n_points=sp, h0=-48.0,h1=24+48,sort_by_length=False)
            
            error_cov,range_0=linearized_errors(o,radar_e3d,all_t_obs,include_drag=True)
            mean_error,last_error,max_error=sample_orbit(o,error_cov,all_t_obs,plot=True,t_obs=t_sample)
            print("n_points %d n_tracklets %d mean_error %f"%(sp, all_n_tracklets, mean_error))



def initial_discovery(beam_width=1.0):
    """
    Simulate the process of object acquisition into a catalog.
    """
    lost_threshold=10e3
    mean_error_threshold=100.0
    
    h=h5py.File("master/drag_info.h5","r")
    radar_e3d = rl.eiscat_3d(beam='interp', stage=1)

    # space object population
    pop_e3d = plib.filtered_master_catalog_factor(
        radar = radar_e3d,   
        treshhold = 0.01,    # min diam
        min_inc = 50,
        prop_time = 48.0,
    )
    t=n.linspace(0,24*3600,num=1000)

    mean_error_l=[]
    oid_l=[]
    n_tracklets_l=[]
    n_tracklets_per_day_l=[]
    n_points_l=[]

    o_idx_c=0
    objs=[]

    okay_ids=[]
    okay_errs=[]    
    all_ids=[]
    
    for o in pop_e3d.object_generator():
        objs.append(o)
    for oi in range(comm.rank,len(objs),comm.size):
        o=objs[oi]
        all_ids.append(o.oid)
        # get drag info
        idx=n.where(h["oid"].value == o.oid)[0]
        o.alpha=h["alpha"].value[idx]
        o.t1=h["t1"].value[idx]
        o.beta=h["beta"].value[idx]                
        n_tracklets = 0
        # get measurement times


        
        try:
            all_t_obs,all_n_tracklets,all_t_means,all_t_lens=get_t_obs(o, radar_e3d, n_tracklets=24, track_length=3600.0, n_points=1, h0=0.0,h1=48,sort_by_length=False)

            ho=h5py.File("iods/iods_info_%02d.h5"%(comm.rank),"w")
            ho["okay"]=okay_ids
            ho["okay_errs"]=okay_errs
            ho["all"]=all_ids        
            ho.close()
        
        
            for ti,t0 in enumerate(all_t_means):
                print("initial discovery simulation oid %d, iod %d"%(o.oid,ti))

                t_obs,n_tracklets,t_means,t_lens=get_t_obs(o, radar_e3d, n_tracklets=1, track_length=3600.0, n_points=1, h0=t0/3600 - 1.0,h1=49,sort_by_length=False)
                initial_t0=t_obs[0]
                # change epoch to discovery, so that at detection t=0
                o2=change_of_epoch_test(o,t_obs[0],plot=False)
                o2.alpha=o.alpha
                o2.t1=o.t1
                o2.beta=o.beta
                o2.oid=o.oid
                # drag not important for a < 10 minute interval
                error_cov,range_0=linearized_errors(o2,radar_e3d,t_obs-initial_t0,include_drag=False)
                lost_threshold = range_0*n.sin(n.pi*beam_width/180.0)*1e3
                
                print("range_0 %f km lost_threshold %f (m)"%(range_0,lost_threshold))
                

                mean_error,last_error,max_error=sample_orbit(o2,error_cov,t_means-initial_t0,plot_t_means=t_obs-initial_t0,plot=True,t_obs=n.linspace(0,t_lens[0],num=200),use_atmospheric_errors=False,fname="iods/iod_%d_%03d_00.png"%(o.oid,ti),max_error=lost_threshold)

                if last_error > lost_threshold:
                    print("object lost")
                    continue
            
                t_obs_next,n_tracklets_next,t_means_next,t_lens_next=get_t_obs(o, radar_e3d, n_tracklets=24, track_length=3600.0, n_points=1, h0=initial_t0/3600.0+1.0,h1=48+initial_t0/3600.0,sort_by_length=False, n_points0=50)

                for nti in range(n_tracklets_next):

                    next_pass_hour=(t_obs_next[nti]-initial_t0)/3600.0
        
                    t_obs,n_tracklets,t_means,t_lens=get_t_obs(o, radar_e3d, n_tracklets=nti+1, track_length=3600.0, n_points=10, h0=initial_t0/3600.0-1.0,h1=49+initial_t0/3600.0, half_track=True,sort_by_length=False, n_points0=50)

                    error_cov,range_0=linearized_errors(o2,radar_e3d,t_obs-initial_t0,include_drag=True)
                    mean_error,last_error,max_error=sample_orbit(o2,error_cov,t_means-initial_t0,plot_t_means=t_obs-initial_t0,plot=True,t_obs=n.linspace(0,next_pass_hour*3600.0,num=1000),fname="iods/iod_%d_%03d_%02d.png"%(o.oid,ti,nti+1),max_error=lost_threshold,use_atmospheric_errors=True)
                    print("%d th pass %f mean_error %f"%(nti+1,next_pass_hour,mean_error))
                    if last_error > lost_threshold:
                        print("object lost")
                        break
                print(nti)
                if nti == (n_tracklets_next - 1) and (n_tracklets_next > 2):
                    okay_ids.append(o.oid)
                    okay_errs.append(mean_error)
                    print("object acquired")
                    break
        except:
            print("problem.")
            pass
        
            
            



def catalogue_maintenance(mean_error_threshold=100.0):
    """ 
    figure out:
    - what are times of observations of objects
    - how many pulses are needed in order to maintain the objects
    - store results
    
    This can be used to "design" measurements for catalog objects.
    """
    h=h5py.File("master/drag_info.h5","r")
    
    radar_e3d = rl.eiscat_3d(beam='interp', stage=1)

    # space object population
    pop_e3d = plib.filtered_master_catalog_factor(
        radar = radar_e3d,   
        treshhold = 0.01,    # min diam
        min_inc = 50,
        prop_time = 48.0,
    )
    objs=[]
    for o in pop_e3d.object_generator():
        objs.append(o)

    all_ids=[]
    t=n.linspace(-12*3600,(24+12)*3600,num=1000)
    for oi in range(comm.rank,len(objs),comm.size):
        o=objs[oi]
        all_ids.append(o.oid)
        # get drag info
        idx=n.where(h["oid"].value == o.oid)[0]
        o.alpha=h["alpha"].value[idx]
        o.t1=h["t1"].value[idx]
        o.beta=h["beta"].value[idx]                

        for n_points in [1,3,10]:
            t_obs,n_tracklets,t_means,t_lens=get_t_obs(o, radar_e3d, n_tracklets=100, track_length=3600.0, n_points=n_points, h0=-12,h1=24+12,sort_by_length=True)
            
            error_cov,range_0=linearized_errors(o,radar_e3d,t_obs,include_drag=True)
            mean_error,last_error,max_error=sample_orbit(o,
                                                         error_cov,
                                                         t_means,
                                                         plot_t_means=t_obs,
                                                         plot=True,
                                                         use_atmospheric_errors=True,
                                                         save_if_okay=False,
                                                         fname="tracks/track-%d.png"%(o.oid),
                                                         mean_error_threshold=mean_error_threshold,
                                                         t_obs=t)
            
            print("rank %d oid %d a %1.0f n_passes per day %f n_points %d mean_error %f (m)"%(comm.rank,o.oid,o.a,n_tracklets/2.0,n_points,mean_error))           
            
            ho=h5py.File("tracks/track-%d.h5"%(o.oid),"w")
            ho["t_obs"]=t_obs
            ho["n_points"]=n_points
            ho["n_tracklets"]=n_tracklets
            ho["t_means"]=t_means
            ho["t_lens"]=t_lens
            ho["mean_error"]=mean_error
            ho["last_error"]=last_error
            ho["max_error"]=max_error                
            ho.close()
            if mean_error < mean_error_threshold:
                break
                
                


def example_err_cov():
    radar_e3d = rl.eiscat_3d(beam='interp', stage=1)    
    o=so.SpaceObject(a=7000.0,e=1e-3,i=69,raan=89,aop=12,mu0=47,d=1.0,A=1.0,m=1.0)
    
    t_obs,n_tracklets,t_means,t_lens=get_t_obs(o, radar_e3d, n_tracklets=1, track_length=3600.0, n_points=3, h0=0,h1=24,sort_by_length=True)

    o1=change_of_epoch_test(o,t_obs[0],plot=False)
    o1.alpha=4.1
    o1.beta=7.8
    o1.t1=8943.65

    t_obs,n_tracklets,t_means,t_lens=get_t_obs(o1, radar_e3d, n_tracklets=1, track_length=3600.0, n_points=3, h0=-1,h1=1,sort_by_length=True)    
    
    error_cov,range_0=linearized_errors(o1,radar_e3d,t_obs,include_drag=True)

    mean_error,last_error=sample_orbit(o1,
                                       error_cov,
                                       t_obs,              # mean measurement times
                                       plot_t_means=t_obs, # where to plot lines for measurement times
                                       plot=True,
                                       t_obs=n.linspace(-12*3600,12*3600,num=1000),
                                       use_atmospheric_errors=False,
                                       fname="report_plots/err_cov_ex1.png")

    t_obs,n_tracklets,t_means,t_lens=get_t_obs(o1, radar_e3d, n_tracklets=2, track_length=3600.0, n_points=3, h0=-12,h1=12,sort_by_length=True)
    error_cov,range_0=linearized_errors(o1,radar_e3d,t_obs,include_drag=True)

    mean_error,last_error=sample_orbit(o1,
                                       error_cov,
                                       t_means,            # mean measurement times 
                                       plot_t_means=t_obs, # where to plot lines for measurement times
                                       plot=True,
                                       t_obs=n.linspace(-12*3600,12*3600,num=1000),
                                       use_atmospheric_errors=False,
                                       fname="report_plots/err_cov_ex2.png")
    
    t_obs,n_tracklets,t_means,t_lens=get_t_obs(o1, radar_e3d, n_tracklets=10, track_length=3600.0, n_points=3, h0=-12,h1=12,sort_by_length=True)
    error_cov,range_0=linearized_errors(o1,radar_e3d,t_obs,include_drag=True)

    mean_error,last_error=sample_orbit(o1,
                                       error_cov,
                                       t_means,
                                       plot_t_means=t_obs,
                                       plot=True,
                                       t_obs=n.linspace(-12*3600,12*3600,num=1000),
                                       use_atmospheric_errors=False,
                                       fname="report_plots/err_cov_ex3.png")

    mean_error,last_error=sample_orbit(o1,error_cov,t_means,plot_t_means=t_obs,plot=True,t_obs=n.linspace(-96*3600,96*3600,num=1000),use_atmospheric_errors=True,fname="report_plots/err_cov_ex4.png")    

def example_err_cov2():
    radar_e3d = rl.eiscat_3d(beam='interp', stage=1)    
#    o=so.SpaceObject(a=7000.0,e=1e-3,i=69,raan=89,aop=12,mu0=47,d=1.0,A=1.0,m=1.0)
    o=so.SpaceObject(a=26571.100000,e=0.606000, i=69.320000,d=3.154000, A=7.813436, m=1239.000000, raan=89,aop=12,mu0=47,oid=43)

    tau,beta,t1,alpha=ade.atmospheric_errors(o,a_err_std=0.05,N_samps=200,plot=False,threshold_error=100.0)
    o.alpha=alpha
    o.beta=beta
    o.t1=t1

    for n_points in [1,3,10,100]:
        t_obs,n_tracklets,t_means,t_lens=get_t_obs(o, radar_e3d, n_tracklets=50, track_length=3600.0, n_points=n_points, h0=-48,h1=24+48,sort_by_length=False)    
        error_cov,range_0=linearized_errors(o,radar_e3d,t_obs,include_drag=True)
        
        idx=n.where((t_means > -24*3600) & (t_means < 48*3600))[0]
        idx2=n.where((t_obs > -24*3600) & (t_obs < 48*3600))[0]        
        
        mean_error,last_error=sample_orbit(o,
                                           error_cov,
                                           t_means[idx],
                                           plot_t_means=t_obs[idx2],
                                           plot=True,
                                           t_obs=n.linspace(-24*3600,48*3600,num=1000),
                                           use_atmospheric_errors=True,
                                           fname="report_plots/main_%d_%04d_%d.png"%(o.oid,n_points,n.log10(o.A/o.m)))


if __name__ == "__main__":
    catalogue_maintenance()
#    initial_discovery()    
#    example_err_cov2()
#    example_err_cov()    
#    initial_discovery()    
#    error_sweeps()
#    initial_discovery()
