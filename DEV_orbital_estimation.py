import numpy as n

import orbital_estimation as oe
import population_library as plib
import radar_library as rlib
import simulate_tracking
import simulate_tracklet as st
import os
import h5py
import re
import glob
import space_object as spo
import scipy.optimize as so
import matplotlib.pyplot as plt 
import dpt_tools as dpt
from pandas.plotting import scatter_matrix
import pandas as pd
import seaborn as sb

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
 Some orbital determination algorithms. 

 Juha Vierinen
"""

# simulate measurements for od
def sim_meas(o,m_time,rx_locs,tx_locs, dt=0.1):
    """
    Simulate a measurement for a tx-rx pair
    """
    n_tracklets=len(m_time)
    m_ranges=[]
    m_range_rates=[]

    for ti in range(n_tracklets):

        ecefs=o.get_orbit(m_time[ti])

        ecefs_p=o.get_orbit(m_time[ti]+dt)
        ecefs_m=o.get_orbit(m_time[ti]-dt)

        ranges0=n.sqrt((ecefs[0,:]-tx_locs[ti,0])**2.0+(ecefs[1,:]-tx_locs[ti,1])**2.0+(ecefs[2,:]-tx_locs[ti,2])**2.0)
        ranges1=n.sqrt((ecefs[0,:]-rx_locs[ti,0])**2.0+(ecefs[1,:]-rx_locs[ti,1])**2.0+(ecefs[2,:]-rx_locs[ti,2])**2.0)   
        ranges=ranges0+ranges1


        ranges0=n.sqrt((ecefs_p[0,:]-tx_locs[ti,0])**2.0+(ecefs_p[1,:]-tx_locs[ti,1])**2.0+(ecefs_p[2,:]-tx_locs[ti,2])**2.0)
        ranges1=n.sqrt((ecefs_p[0,:]-rx_locs[ti,0])**2.0+(ecefs_p[1,:]-rx_locs[ti,1])**2.0+(ecefs_p[2,:]-rx_locs[ti,2])**2.0)   
        ranges_p=ranges0+ranges1

        ranges0=n.sqrt((ecefs_m[0,:]-tx_locs[ti,0])**2.0+(ecefs_m[1,:]-tx_locs[ti,1])**2.0+(ecefs_m[2,:]-tx_locs[ti,2])**2.0)
        ranges1=n.sqrt((ecefs_m[0,:]-rx_locs[ti,0])**2.0+(ecefs_m[1,:]-rx_locs[ti,1])**2.0+(ecefs_m[2,:]-rx_locs[ti,2])**2.0)   
        ranges_m=ranges0+ranges1

        range_rates=0.5*(ranges_p-ranges)/dt + 0.5*(ranges-ranges_m)/dt
        m_ranges.append(ranges/1e3)
        m_range_rates.append(range_rates/1e3)        
    return(m_ranges, m_range_rates)



# simulate measurements for od, used to test if od works or not
def get_states(o, m_time, dt=0.1):
    """
    Get state of object at times m_time
    """
    ecefs=o.get_orbit(m_time)
    ecefs_p=o.get_orbit(m_time+dt)
    ecefs_m=o.get_orbit(m_time-dt)
    vels=0.5*(ecefs_p-ecefs)/dt + 0.5*(ecefs-ecefs_m)/dt

    states=n.zeros([6,len(m_time)])
    states[0:3,:]=ecefs
    states[3:6,:]=vels
    return(states)



# compare two state vectors
# can be used to compare with "ground truth"
def compare_states(o1,o2,m_time,c_time_max=48*3600, plot=False):
    """
    Compare the state vectors of two space objects. Plot ecef positions and position differences.
    """
    c_time_max = n.max([c_time_max,n.max(m_time[len(m_time)-1])])
    c_time=n.linspace(-c_time_max,c_time_max,num=1000)
    # best fit states
    states=get_states(o1, c_time)
    # "true" states
    true_states=get_states(o2, c_time)

    range_resid=n.sqrt( (true_states[0,:]-states[0,:])**2.0 + (true_states[1,:]-states[1,:])**2.0 + (true_states[2,:]-states[2,:])**2.0 )
    vel_resid=n.sqrt( (true_states[0+3,:]-states[0+3,:])**2.0+(true_states[1+3,:]-states[1+3,:])**2.0+(true_states[2+3,:]-states[2+3,:])**2.0 )
    
    if plot:
        plt.subplot(121)        
        plt.plot(c_time/3600.0, range_resid )
        plt.ylabel("ECEF position residual (m)")
        plt.xlabel("Time (h)")

        for i in range(len(m_time)):
            for j in range(len(m_time[i])):
                plt.axvline(m_time[i][j]/3600.0)
    
        plt.subplot(122)

        plt.plot(c_time/3600.0, vel_resid)
        plt.ylabel("ECEF velocity residual (m/s)")
        plt.xlabel("Time (h)")

        for i in range(len(m_time)):
            for j in range(len(m_time[i])):
                plt.axvline(m_time[i][j]/3600.0)
    
        plt.show()
    return(c_time, range_resid, vel_resid)
    
    
def mcmc_od(m_time, m_range, m_range_rate, m_range_std, m_range_rate_std, tx_locs, rx_locs, o_prior, dt=0.1,
            method="Nelder-Mead",
            sigma_a=1000.0,      # very non-informative priors for the orbital elements
            sigma_e=10.0,        # means based on o_prior
            sigma_i=100.0,
            sigma_raan=100.0,
            sigma_aop=100.0,
            sigma_mu0=100.0,
            sigma_am=0.77,      # log10(A/m) stdev, default based on all master catalog objects
            am_mean=-0.5,       # log10(A/m) mean, default based on all master catalog objects
            N_samples=5000, mcmc=False, thinning=10, odir="./test_tracklets", plot=False):

    """
    fmin search and mcmc based OD using weighted linear-least squares.

    The MCMC algorithm uses the Single Component Adaptive Metropolis-hastings (SCAM) algorithm,
    which is relatively robust, and somewhat efficient. This could be in the future 
    augmented with online estimation of proposal distribution covriance from initial samples 
    to improve sampling efficiency. 
    """

    tx_locs=n.array(tx_locs)
    rx_locs=n.array(rx_locs)
    
    n_tracklets=len(m_time)
    
    print("n_tracklets %d"%(n_tracklets))

    s_ranges, s_range_rates=sim_meas(o_prior,m_time,rx_locs,tx_locs)
    print("range residuals")
    for i in range(n_tracklets):
        print(s_ranges[i]-m_range[i])
    print("range_rate residuals")
    for i in range(n_tracklets):    
        print(s_range_rates[i]-m_range_rate[i])

    A_to_m=o_prior.A/o_prior.m

    # don't start with real values to make map estimate realistic
    x0=n.array([o_prior.a+n.random.randn(1)*0.0,
                n.log10(o_prior.e)+n.random.randn(1)*0.0,
                o_prior.i+n.random.randn(1)*0.0,
                o_prior.raan+n.random.randn(1)*0.0,
                o_prior.aop+n.random.randn(1)*0.0,
                o_prior.mu0+n.random.randn(1)*0.0,
                n.log10(A_to_m)+n.random.randn(1)*0.1])
    
    xtrue=n.array([o_prior.a,
                   n.log10(o_prior.e),
                   o_prior.i,
                   o_prior.raan,
                   o_prior.aop,
                   o_prior.mu0,
                   n.log10(A_to_m)])

    # object used for least squares c
    
    def ss(x):
        a=x[0]
        e=10**(x[1])
        i=x[2]
        raan=x[3]
        aop=x[4]
        mu0=x[5]
        A_to_m=10**(x[6])

        if x[6] < -4.0 or x[6] > 100.0:
            return(1e6)
        o=spo.SpaceObject(a=a, e=e, i=i, raan=raan, aop=aop, mu0=mu0, A=A_to_m, m=1.0, mjd0=o_prior.mjd0)        
        # forward model
        s_range, s_range_rate=sim_meas(o,m_time,rx_locs,tx_locs)

        s=0.0
        for i in range(n_tracklets):
            s+=0.5*n.sum( (n.abs(m_range[i]-s_range[i])**2.0)/(m_range_std[i]**2.0) + n.sum( (n.abs(m_range_rate[i]-s_range_rate[i])**2.0)/(m_range_rate_std[i]**2.0)) )
            
        # Gaussian prior for log10(A/m) mean = -0.5, std=0.77, based on the
        # approximate distribution of log10(A/m) in the MASTER 2009 catalog
        s += 0.5*n.sum( (x[6] + 0.5)**2.0/0.77**2.0 )
        # Gaussian prior distribution for Keplerian parameters
        s += 0.5*n.sum( (x[0] - x0[0])**2.0/sigma_a**2.0 )
        s += 0.5*n.sum( (x[1] - x0[1])**2.0/sigma_e**2.0 )
        s += 0.5*n.sum( (x[2] - x0[2])**2.0/sigma_i**2.0 )
        s += 0.5*n.sum( (x[3] - x0[3])**2.0/sigma_raan**2.0 )
        s += 0.5*n.sum( (x[4] - x0[4])**2.0/sigma_aop**2.0 )
        s += 0.5*n.sum( (x[5] - x0[5])**2.0/sigma_mu0**2.0 )                    
        print(s)
        return(s)

    print("maximum a posteriori search...")
    
    xhat=so.minimize(ss, x0, method=method).x
#    xhat=so.fmin(ss,x0,full_output=False,disp=True)

    print(xhat)
    print(xhat-xtrue)
    
    o_fit=spo.SpaceObject(a=xhat[0],e=10**xhat[1],i=xhat[2],raan=xhat[3],aop=xhat[4],mu0=xhat[5],A=10**xhat[6],m=1.0, mjd0=o_prior.mjd0)
    best_fit_ranges, best_fit_range_rates=sim_meas(o_fit,m_time,rx_locs,tx_locs)

    if rank==0:
        if plot:
            plt.subplot(121)
            for i in range(n_tracklets):        
                plt.plot(m_time[i], 1e3*(best_fit_ranges[i]-m_range[i]),"o",label="tracklet %d"%(i))
            plt.xlabel("Time (s)")
            plt.ylabel("Range residual (m)")
        
            plt.subplot(122)
            for i in range(n_tracklets):                    
                plt.plot(m_time[i], 1e3*(best_fit_range_rates[i]-m_range_rate[i]),"o",label="tracklet %d"%(i))
            plt.xlabel("Time (s)")
            plt.ylabel("Range-rate residual (m/s)")        
            plt.show()
        #
        s_time, range_resid, vel_resid = compare_states(o_prior,o_fit,m_time,plot=plot)
    comm.Barrier()
        
    if not mcmc:
        return(o_fit, s_time, range_resid, vel_resid, m_time)

    print("mcmc sampling...")    
    # Single Component Adaptive Metropolis-hastings MCMC (SCAM) 
    chain=n.zeros([N_samples, len(x0)])
    xnow=n.copy(xhat)
    logss=-ss(xnow)   # log likelihood
    n_par=len(xhat)
    
    step=n.array([1e-4, 0.002, 0.5e-4, 1.5e-4, 1e-4, 1e-4, 0.01])*1.0
    accept=n.zeros(len(step))
    tries=n.zeros(len(step))    
    for i in range(N_samples*thinning):
        xtry=n.copy(xnow)
        pi=int(n.floor(n.random.rand(1)*n_par))
        xtry[pi]+=n.random.randn(1)*step[pi]
        logss_try=-ss(xtry)  # log likelihood
        alpha = n.log(n.random.rand(1))
        
        # proposal step
        # accept always
        if logss_try > logss:
            logss=logss_try
            xnow=xtry
            accept[pi]+=1
        # accept by random chance
        elif (logss_try - alpha) > logss:
            logss=logss_try
            xnow=xtry
            accept[pi]+=1
        #
        # TBD: Use DRAM to speed up sampling, because parameters
        # TBD2: use linearized error estimate to come up with cov matrix for proposal
        # are highly correlated.
        # After sufficiently many samples drawn with SCAM
        # estimate covariance with C=n.cov(n.transpose(chain))
        # use n.random.multivariate_normal to sample from C
        #
        chain[int(i/thinning),:]=xnow
        tries[pi]+=1.0

        # adaptive step scam. adjust proposal
        # based on acceptance probability
        if i%100 == 0 and i>0:
            print("step %d/%d"%(i,N_samples*thinning))
            print("logss %1.2f current state"%(logss))
            print(xnow)
            print("acceptance probability")            
            print(accept/tries)
            print("step size")                        
            print(step)
            print("---")      
            ratio=accept/tries
            too_many=n.where(ratio > 0.5)[0]
            too_few=n.where(ratio < 0.3)[0]
            step[too_many]=step[too_many]*2.0
            step[too_few]=step[too_few]/2.0
            accept[:]=0.0
            tries[:]=0.0
        
        if i%100 == 0 and i>0:
            print("saving")
            ho=h5py.File("%s/chain-%03d.h5"%(odir,rank),"w")
            ho["chain"]=chain[0:(int(i/thinning)),:]
            ho.close()
    
    return(o_fit)

def plot_chain(oid=8):

    dname="test_tracklets_%d"%(oid)
    
    m = plib.master_catalog(sort=False)
    o = m.get_object(oid)

    fl=glob.glob("%s/chain*.h5"%(dname))

    chains=[]
    lens=[]
    for fi,f in enumerate(fl):
        print(f)
        ho=h5py.File(f,"r")
        c=n.copy(ho["chain"].value)
        clen=c.shape[0]
        chains.append(c[(clen/2):clen,:])
#        chains.append(c)
        
        lens.append(c.shape[0])
        ho.close()
    c=n.vstack(chains)
    
    print(c.shape)

    d={"a":c[:,0],
       "log10(e)":c[:,1],
       "i":c[:,2],
       "raan":c[:,3],
       "aop":c[:,4],
       "mu0":c[:,5],
       "log10(A/m)":c[:,6]
    }

    print(" a %f +/- %f\n e %f +/- %f\n i %f +/- %f\n raan %f +/- %f\n aop %f +/- %f\n mu0 %f +/- %f\n log10(A/m) %f +/- %f\n"%(n.mean(c[:,0]),n.std(c[:,0]),
                                                                                                                                n.mean(c[:,1]),n.std(c[:,1]),
                                                                                                                                n.mean(c[:,2]),n.std(c[:,2]),
                                                                                                                                n.mean(c[:,3]),n.std(c[:,3]),
                                                                                                                                n.mean(c[:,4]),n.std(c[:,4]),
                                                                                                                                n.mean(c[:,5]),n.std(c[:,5]),
                                                                                                                                n.mean(c[:,6]),n.std(c[:,6])))
    
    df = pd.DataFrame(d)

    pairs=[["a","log10(e)"],
           ["a","i"],
           ["a","mu0"],
           ["a","aop"],
           ["a","raan"],
           ["a","log10(A/m)"],
           ["aop","mu0"],
           ["log10(e)","i"]]

    true_vals={"a":o.a,
               "log10(e)":n.log10(o.e),
               "i":o.i,
               "raan":o.raan,
               "aop":o.aop,
               "mu0":o.mu0,
               "log10(A/m)":n.log10(o.A/o.m)}

    for p in pairs:
        g = sb.jointplot(x=p[0], y=p[1], data=df, kind="hex")
#        g.plot_joint(plt.scatter, c="b", s=30, linewidth=1, marker="+", alpha=0.05)
#        plt.hist2d(df[p[0]],df[p[1]],bins=100)
        plt.xlabel(p[0])
        plt.ylabel(p[1])        
        plt.axvline(true_vals[p[0]])
        plt.axhline(true_vals[p[1]])        
        plt.show()
    
#    sb.pairplot(df)
 #   plt.show()

    
    

# Unit test
#
# Use MCMC to estimate a state vector
#
def wls_state_est(mcmc=False, n_tracklets=1, track_length=600.0, n_points=3, oidx=145128, N_samples=5000, plot=True, o=None):
    """
    Weighted linear least squares estimation of orbital elements
    
    Simulate measurements using create tracklet and estimate 
    orbital parameters, which include six keplerian and area to mass ratio. 

    Use fmin search. 
    Optionally utilize MCMC to sample the distribution of parameters.
    
    number of tracklets, tracklet length, and number of tracklet points per tracklet are
    user definable, allowing one to try out different measurement strategies. 
    """
    # first we shall simulate some measurement
    # Envisat

    if o == None:
        m = plib.master_catalog(sort=False)
        o = m.get_object(oidx)
        oid=o.oid
    else:
        oid=42

    dname="./test_tracklets_%d_nt_%d_np_%d"%(oid,n_tracklets,n_points)

    print(o)

    # figure out epoch in unix seconds
    t0_unix = dpt.jd_to_unix(dpt.mjd_to_jd(o.mjd0))

    if rank == 0:
        os.system("rm -Rf %s"%(dname))
        os.system("mkdir %s"%(dname))    
        
        e3d = rlib.eiscat_3d(beam='gauss')

        # time in seconds after mjd0
        t_all = n.linspace(0, 2*24*3600, num=100000)
    
        passes, _, _, _, _ = simulate_tracking.find_pass_interval(t_all, o, e3d)
        print(passes)
        if n_tracklets == None:
            n_tracklets = len(passes[0])
        n_tracklets = n.min([n_tracklets,len(passes[0])])

        tracklet_idx = n.arange(n_tracklets)
        n.random.shuffle(tracklet_idx)
        for pi in tracklet_idx:
            p = passes[0][pi]
            mean_t=0.5*(p[1]+p[0])
            print("duration %1.2f"%(p[1]-p[0]))
            if p[1]-p[0] > 5.0:
                if n_points == 1:
                    t_obs=n.array([mean_t])
                else:
                    t_obs=n.linspace(n.max([p[0],mean_t-track_length/2]), n.min([p[1],mean_t+track_length/2]),num=n_points)
                
                print(t_obs)
                meas, fnames, ecef_stdevs = st.create_tracklet(o, e3d, t_obs, hdf5_out=True, ccsds_out=True, dname=dname)

    # then we read these measurements
    comm.Barrier()
    
    fl=glob.glob("%s/*"%(dname))
    for f in fl:
        print(f)
        fl2=glob.glob("%s/*.h5"%(f))
        print(fl2)
        fl2.sort()
        
        true_states=[]
        all_r_meas=[]
        all_rr_meas=[]
        all_t_meas=[]
        all_true_states=[]
        tx_locs=[]
        rx_locs=[]
        range_stds=[]
        range_rate_stds=[]        
        
        for mf in fl2:
            h=h5py.File(mf,"r")
            all_r_meas.append(n.copy(h["m_range"].value))
            all_rr_meas.append(n.copy(h["m_range_rate"].value))
            all_t_meas.append(n.copy(h["m_time"].value-t0_unix))
            all_true_states.append(n.copy(h["true_state"].value))
            tx_locs.append(n.copy(h["tx_loc"].value))
            rx_locs.append(n.copy(h["rx_loc"].value))
            range_stds.append(n.copy(h["m_range_rate_std"].value))
            range_rate_stds.append(h["m_range_std"].value)
            h.close()
            
        
        # get best fit space object
        return(mcmc_od(all_t_meas, all_r_meas, all_rr_meas, range_stds, range_rate_stds, tx_locs, rx_locs, o, mcmc=mcmc, odir=dname, N_samples=N_samples, plot=plot))

def sweep_n_tracklets():
    
    wls_state_est(mcmc=False, oidx=8, n_tracklets=2, track_length=600.0, n_points=3, N_samples=50000)
    wls_state_est(mcmc=False, oidx=8, n_tracklets=4, track_length=600.0, n_points=3, N_samples=50000)
    wls_state_est(mcmc=False, oidx=8, n_tracklets=8, track_length=600.0, n_points=3, N_samples=50000)
    wls_state_est(mcmc=False, oidx=8, n_tracklets=16, track_length=600.0, n_points=3, N_samples=50000)
    wls_state_est(mcmc=False, oidx=8, n_tracklets=32, track_length=600.0, n_points=3, N_samples=50000)            
    
def sweep_n_points():
    
    wls_state_est(mcmc=False, oidx=8, n_tracklets=1, track_length=600.0, n_points=16, N_samples=50000)
    wls_state_est(mcmc=False, oidx=8, n_tracklets=1, track_length=600.0, n_points=32, N_samples=50000)
    wls_state_est(mcmc=False, oidx=8, n_tracklets=1, track_length=600.0, n_points=64, N_samples=50000)
    wls_state_est(mcmc=False, oidx=8, n_tracklets=1, track_length=600.0, n_points=128, N_samples=50000)
    wls_state_est(mcmc=False, oidx=8, n_tracklets=1, track_length=600.0, n_points=256, N_samples=50000)
    wls_state_est(mcmc=False, oidx=8, n_tracklets=1, track_length=600.0, n_points=512, N_samples=50000)
        
# use fmin search
if __name__ == "__main__":
    o=spo.SpaceObject(a=6770.0, e=1e-4, i=89.0, raan=12.0, aop=23.0, mu0=32.0, A=10**(-2.0), m=1.0, d=1.0)        
    
    wls_state_est(mcmc=False, o=o, n_tracklets=3, track_length=1000.0, n_points=10, N_samples=50000)
#    wls_state_est(mcmc=False, oidx=8, n_tracklets=16, track_length=600.0, n_points=10, N_samples=50000)    
#    initial_iod_accuracy()
