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

def get_t_obs(o, radar, n_tracklets=1, track_length=600.0, n_points=3, debug=True,h0=0.0,h1=24.0):
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
    # seems like random shuffle doesn't use a random seed.
    import time
    n.random.seed(int(time.time()))
    n.random.shuffle(tracklet_idx)
    
    n_tracklets = n.min([n_tracklets,len(passes[0])])
    tracklet_idx=tracklet_idx[0:n_tracklets]
    
    t_obss=[]
    t_means=[]
    for pi in tracklet_idx:
        p = passes[0][pi]
        mean_t=0.5*(p[1]+p[0])
        t_means.append(mean_t)
        if debug:
            print("duration %1.2f"%(p[1]-p[0]))
        if p[1]-p[0] > 5.0:
            if n_points == 1:
                t_obs=n.array([mean_t])
            else:
                # maximize track length
                t_obs=n.linspace(n.max([p[0],mean_t-track_length/2]), n.min([p[1],mean_t+track_length/2]),num=n_points)
            t_obss=n.concatenate([t_obss,t_obs])
    return(t_obss,n_tracklets,n.array(t_means))


def linearized_errors(ot,radar,t_obs,plot=True,debug=False,t0s=n.linspace(0,2*3600,num=50),
                      debug_output=True,
                      time_vector=False, dx=0.0001, dv=1e-6, dcd=0.01, include_drag=False):
    """ 
    Use numerical differences to estimate the measurement error covariance matrix using
    cartesian state vector at epoch plus area-to-mass ratio (7 orbital parameters)
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
    for txi in range(n_tx):
        for rxi in range(n_rx):
            n_meas0=len(meas0[txi][rxi]["m_time"])
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

    # drag prior
    # Virtual measurement: we've measured log10(A/m) with 0.77 standard deviation
    # doing Bayesian statistics with linear regression :)
    if include_drag:
        J[n_meas,6]=1.0
        Sigma_m_inv[n_meas,n_meas]=1.0/0.77**2.0
        
    # linearized error covariance
    # \Sigma_post = (A^T \Sigma^{-1} A)^{-1} 
    Sigma_pos=n.linalg.inv(n.dot(n.dot(n.transpose(J),Sigma_m_inv),J))

    if debug_output:
        print("ECEF error stdev")
        print(n.sqrt(n.diag(Sigma_pos))[0:6]*1e3)
        if include_drag:
            print("Drag coefficient error")    
            print(n.sqrt(n.diag(Sigma_pos))[6])
    
    return(Sigma_pos)


def sample_orbit(o,C,t_means,N=100,alpha=4.1,beta=7.8,t1=8943.65):
    t_obs=n.linspace(-8*24*3600,8*24*3600,num=1000)

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

    atmos_err_stdev=ad.atmospheric_errors_rw(t_means, t=t_obs, alpha=o.alpha, beta=o.beta, t1=o.t1, error0=1.0,loglog=False, show_model=False,plot=False)

    total_err_stdev=n.sqrt(atmos_err_stdev**2.0 + diff2/N)
    mean_error=n.mean(total_err_stdev[t_idx])
    
    plt.semilogy(t_obs/3600,n.sqrt(diff2/N),label="orbit error")
    plt.semilogy(t_obs/3600,atmos_err_stdev,label="atmos error")
    plt.semilogy(t_obs/3600,total_err_stdev,label="total error")
    plt.legend()
    for t in t_means:
        plt.axvline(t/3600.0,color="C3")
    plt.title("oid=%d a=%1.0f km log$_{10}$(e)=%1.1f\n mean error %1.1f (m) i=%1.0f d=%1.2f log$_{10}(A/m)=%1.1f$"%(o.oid,o.a,n.log10(o.e),mean_error,o.i,o.d,n.log10(o.A/o.m)))
    plt.xlabel("Time (h)")
    plt.ylabel("Position error standard deviation (m)")    
    plt.show()

if __name__ == "__main__":

    h=h5py.File("master/drag_info.h5","r")
    dx=0.1
    dvx=0.01
    dc=0.01
    radar_e3d = rl.eiscat_3d(beam='interp', stage=1)

    # space object population
    pop_e3d = plib.filtered_master_catalog_factor(
        radar = radar_e3d,   
        treshhold = 0.01,    # min diam
        min_inc = 50,
        prop_time = 48.0,
    )
    t=n.linspace(0,24*3600,num=1000)

    for o in pop_e3d.object_generator():
        # get drag info
        idx=n.where(h["oid"].value == o.oid)[0]
        o.alpha=h["alpha"].value[idx]
        o.t1=h["t1"].value[idx]
        o.beta=h["beta"].value[idx]                

        t_obs,n_tracklets,t_means=get_t_obs(o, radar_e3d, n_tracklets=20, track_length=3600.0, n_points=1, h0=-12,h1=(24+12))
        # tbd read atmospheric drag uncertain related  diffusions coefficients
        t_mean=n.mean(n.diff(n.sort(t_means)))/3600.0
        print("n_tracklets %d mean diff between measuremetns %1.2f (h) alpha %1.1f beta %1.1f t1 %1.1f"%(n_tracklets,t_mean,o.alpha,o.beta,o.t1))
        error_cov=linearized_errors(o,radar_e3d,t_obs,include_drag=True)
        sample_orbit(o,error_cov,t_means)
        

