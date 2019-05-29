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
import ccsds_write

from sorts_config import p as default_propagator

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
def compare_states(o1,o2):
    """
    Compare the state vectors of two space objects. Plot ecef positions and position differences.
    """
    # best fit states
    m_time=n.linspace(0,24*3600,num=1000)
    states=get_states(o1, m_time)
    # "true" states
    true_states=get_states(o2, m_time)
    plt.subplot(221)
    for i in range(3):
        plt.plot(m_time,true_states[i,:]-states[i,:])
    plt.ylabel("ECEF position residual (m)")
    plt.xlabel("Time (s)")

    plt.subplot(222)
    for i in range(3):
        plt.plot(true_states[i,:])
        plt.plot(states[i,:])
    plt.ylabel("ECEF position (m)")
    plt.xlabel("Time (s)")
        
    plt.subplot(223)        
    for i in range(3):
        plt.plot(true_states[i+3,:]-states[i+3,:])
        
    plt.ylabel("ECEF velocity residual (m/s)")
    plt.xlabel("Time (s)")
        
    plt.subplot(224)                        
    for i in range(3):
        plt.plot(true_states[i+3,:])
        plt.plot(states[i+3,:])
    plt.ylabel("ECEF velocity (m/s)")
    plt.xlabel("Time (s)")
        
    plt.show()
    
    


def mcmc_od(m_time, m_range, m_range_rate, m_range_std, m_range_rate_std, tx_locs, rx_locs, o_prior, dt=0.1,
            N_samples=5000, mcmc=False, thinning=10, odir="./test_tracklets"):
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
    
    x0=n.array([o_prior.a,
                n.log10(o_prior.e),
                o_prior.i,
                o_prior.raan,
                o_prior.aop,
                o_prior.mu0,
                n.log10(A_to_m)])
    
    xtrue=n.array([o_prior.a,
                   n.log10(o_prior.e),
                   o_prior.i,
                   o_prior.raan,
                   o_prior.aop,
                   o_prior.mu0,
                   n.log10(A_to_m)])

    # object used for least squares c

    o_iter = o_prior.copy()
    ofit = o_prior.copy()

    def s_update(x, space_obj):
        space_obj.update(
            a=x[0],
            e=10**(x[1]),
            i=x[2],
            raan=x[3],
            aop=x[4],
            mu0=x[5],
        )
        space_obj.A = 10**(x[6])*space_obj.m

    def ss(x):
        if x[6] < -4.0 or x[6] > 100.0:
            return(1e6)
        
        s_update(x, o_iter)

        # forward model
        s_range, s_range_rate=sim_meas(o_iter,m_time,rx_locs,tx_locs)

        s=0.0
        for i in range(n_tracklets):
            s+=n.sum( (n.abs(m_range[i]-s_range[i])**2.0)/(m_range_std[i]**2.0) + n.sum( (n.abs(m_range_rate[i]-s_range_rate[i])**2.0)/(m_range_rate_std[i]**2.0)) )
        #print(s)
        #print(x)
        #print(o_iter)
        return(0.5*s)

    print("fmin search...")
    xhat=so.fmin(
        ss,
        x0,
        full_output=False,
        disp=True, 
        #maxiter=200,
    )

    
    print(xhat)
    print(xhat-xtrue)

    s_update(xhat, ofit)

    compare_states(o_prior, ofit)

    if not mcmc:
        return(ofit)

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
            print("current state")
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
    
    return(ofit)

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
#        chains.append(c[(clen/2):clen,:])
        chains.append(c)
        
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
        g = sb.jointplot(x=p[0], y=p[1], data=df, kind="kde")
        g.plot_joint(plt.scatter, c="b", s=30, linewidth=1, marker="+", alpha=0.05)
        plt.axvline(true_vals[p[0]])
        plt.axhline(true_vals[p[1]])        
        plt.show()
    
    sb.pairplot(df)
    plt.show()

    
    

# Unit test
#
# Use MCMC to estimate a state vector
#
def wls_state_est(mcmc=False, n_tracklets=1, track_length=600.0, n_points=3, oid=145128, N_samples=5000):
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

    m = plib.master_catalog(sort=False)
    o = m.get_object(oid)

    dname="./test_tracklets_%d"%(oid)
    print(o)

    # figure out epoch in unix seconds
    t0_unix = dpt.jd_to_unix(dpt.mjd_to_jd(o.mjd0))

    if rank == 0:
        os.system("rm -Rf %s"%(dname))
        os.system("mkdir %s"%(dname))    
        
        e3d = rlib.eiscat_3d(beam='gauss')

        # time in seconds after mjd0
        t_all = n.linspace(0, 24*3600, num=1000)
    
        passes, _, _, _, _ = simulate_tracking.find_pass_interval(t_all, o, e3d)
        print(passes)
        if n_tracklets == None:
            n_tracklets = len(passes[0])
        n_tracklets = n.min([n_tracklets,len(passes[0])])

        for pi in range(n_tracklets):
            p = passes[0][pi]
            mean_t=0.5*(p[1]+p[0])
            print("duration %1.2f"%(p[1]-p[0]))
            if p[1]-p[0] > 50.0:
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
            
        # determine orbital elements
        o_prior = m.get_object(oid)

        # get best fit space object
        o_fit=mcmc_od(all_t_meas, all_r_meas, all_rr_meas, range_stds, range_rate_stds, tx_locs, rx_locs, o_prior, mcmc=mcmc, odir=dname, N_samples=N_samples)





# Unit test files
#
# Use MCMC to estimate a state vector
#
def wls_state_est_files(dname, mcmc=False, N_samples=5000, propagator = default_propagator, propagator_options = {}):
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

    raise NotImplementedError()
    
    fl_tdm = glob.glob(dname + "/*.tdm")
    fl_h5 = glob.glob(dname + "/*.h5")
    fl_oem = glob.glob(dname + "/*.oem")

    file_data = []
    for ftdm in fl_tdm:
        data = ftdm.split('/')[-1].split('-')
        file_data.append({
            'unix': float(data[1]),
            'oid': int(data[2]),
            'tx': int(data[3][0]),
            'rx': int(data[3][2]),
        })

    fsort = n.argsort(n.array([x['unix'] for x in file_data])).tolist()

    all_r_meas=[]
    all_rr_meas=[]
    all_t_meas=[]
    all_true_states=[]
    tx_locs=[]
    rx_locs=[]
    range_stds=[]
    range_rate_stds=[]

    prior_data, prior_meta = ccsds_write.read_oem(fl_oem[0])

    x, y, z = prior_data[0]['x'], prior_data[0]['y'], prior_data[0]['z']
    vx, vy, vz = prior_data[0]['vx'], prior_data[0]['vy'], prior_data[0]['vz']

    prior_date = prior_data[0]['date']
    prior_mjd = dpt.npdt2mjd(prior_date)
    #prior_jd = dpt.mjd_to_jd(prior_mjd)

    o_prior = spo.SpaceObject.cartesian(
        x, y, z, vx, vy, vz,
        mjd0=prior_mjd,
        oid=42,
        C_R = 1.0,
        propagator = propagator,
        propagator_options = propagator_options,
    )

    for ind in fsort:

        ftdm, fh5 = fl_tdm[ind], fl_h5[ind]
        obs_data = ccsds_write.read_ccsds(ftdm)

        obs_date = prior_data[0]['date']
        #obs_mjd = dpt.npdt2mjd(obs_date)
        #obs_jd = dpt.mjd_to_jd(obs_mjd)

        h=h5py.File(fh5,"r")
        #all_r_meas.append(n.copy(h["m_range"].value))
        #all_rr_meas.append(n.copy(h["m_range_rate"].value))
        #all_t_meas.append(n.copy(h["m_time"].value-t0_unix))

        all_r_meas.append(obs_data['range']*1e3)
        all_rr_meas.append(obs_data['doppler_instantaneous']*1e3)
        all_t_meas.append( (obs_date - prior_date)/n.timedelta64(1, 's') )

        all_true_states.append(n.copy(h["true_state"].value))
        tx_locs.append(n.copy(h["tx_loc"].value))
        rx_locs.append(n.copy(h["rx_loc"].value))
        range_stds.append(n.copy(h["m_range_rate_std"].value))
        range_rate_stds.append(h["m_range_std"].value)
        h.close()

    # get best fit space object
    o_fit=mcmc_od(all_t_meas, all_r_meas, all_rr_meas, range_stds, range_rate_stds, tx_locs, rx_locs, o_prior, mcmc=mcmc, odir=dname, N_samples=N_samples)


# use fmin search
if __name__ == "__main__":

    # 1. simulate measurements for object #8, one tracklet, tracklet span at most 600 seconds,
    #    3 measurement points in a tracklet
    # 2. fmin search for MAP parameters
    # 3. mcmc sample a posteriori distribution with MPI
    #wls_state_est(mcmc=True, oid=8, n_tracklets=1, track_length=600.0, n_points=1, N_samples=5000)
