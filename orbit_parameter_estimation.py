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
import error_covariance as ec
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
 Some orbital determination algorithms. 

 Juha Vierinen
"""

# simulate measurements for od
def sim_meas(o,m_time,rx_locs,tx_locs, dt=0.01):
    """
    Simulate a measurement for a tx-rx pair
    """
    n_tracklets=len(m_time)
    m_ranges=[]
    m_range_rates=[]

    for ti in range(n_tracklets):

        ecefs=o.get_state(m_time[ti])

#        ecefs0=o.get_orbit(m_time[ti])        
 #       ecefs_p=o.get_orbit(m_time[ti]+dt)
  #      ecefs_m=o.get_orbit(m_time[ti]-dt)

  #      print("ecefs")
   #     print(ecefs[0:3,:])
    #    print("ecefs0")        
     #   print(ecefs0)        

        ranges0=n.sqrt((ecefs[0,:]-tx_locs[ti,0])**2.0+(ecefs[1,:]-tx_locs[ti,1])**2.0+(ecefs[2,:]-tx_locs[ti,2])**2.0)
        ranges1=n.sqrt((ecefs[0,:]-rx_locs[ti,0])**2.0+(ecefs[1,:]-rx_locs[ti,1])**2.0+(ecefs[2,:]-rx_locs[ti,2])**2.0)   
        ranges=ranges0+ranges1

        A=((ecefs[0,:]-tx_locs[ti,0])*ecefs[3,:] + (ecefs[1,:]-tx_locs[ti,1])*ecefs[4,:] + (ecefs[2,:]-tx_locs[ti,2])*ecefs[5,:])/ranges0
        B=((ecefs[0,:]-rx_locs[ti,0])*ecefs[3,:] + (ecefs[1,:]-rx_locs[ti,1])*ecefs[4,:] + (ecefs[2,:]-rx_locs[ti,2])*ecefs[5,:])/ranges1
        range_rates=A+B
        
#        ranges0=n.sqrt((ecefs_p[0,:]-tx_locs[ti,0])**2.0+(ecefs_p[1,:]-tx_locs[ti,1])**2.0+(ecefs_p[2,:]-tx_locs[ti,2])**2.0)
 #       ranges1=n.sqrt((ecefs_p[0,:]-rx_locs[ti,0])**2.0+(ecefs_p[1,:]-rx_locs[ti,1])**2.0+(ecefs_p[2,:]-rx_locs[ti,2])**2.0)   
  #      ranges_p=ranges0+ranges1

#        ranges0=n.sqrt((ecefs_m[0,:]-tx_locs[ti,0])**2.0+(ecefs_m[1,:]-tx_locs[ti,1])**2.0+(ecefs_m[2,:]-tx_locs[ti,2])**2.0)
 #       ranges1=n.sqrt((ecefs_m[0,:]-rx_locs[ti,0])**2.0+(ecefs_m[1,:]-rx_locs[ti,1])**2.0+(ecefs_m[2,:]-rx_locs[ti,2])**2.0)   
  #      ranges_m=ranges0+ranges1
     
#        range_rates2=0.5*(ranges_p-ranges)/dt + 0.5*(ranges-ranges_m)/dt
 #       print("rr1")
  #      print(range_rates)
   #     print("rr2")        
    #    print(range_rates2)        
        
        m_ranges.append(ranges/1e3)
        m_range_rates.append(range_rates/1e3)        
    return(m_ranges, m_range_rates)



# simulate measurements for od, used to test if od works or not
def get_states(o, m_time, dt=0.1):
    """
    Get state of object at times m_time
    """
#    ecefs=o.get_orbit(m_time)
    states=o.get_state(m_time)    
 #   ecefs_p=o.get_orbit(m_time+dt)
  #  ecefs_m=o.get_orbit(m_time-dt)
 #   vels=0.5*(ecefs_p-ecefs)/dt + 0.5*(ecefs-ecefs_m)/dt

  #  states=n.zeros([6,len(m_time)])
  #  states[0:3,:]=ecefs
  #  states[3:6,:]=vels
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
            sigma_x=100.0,      # very non-informative priors for the orbital elements
            sigma_y=100.0,      # very non-informative priors for the orbital elements
            sigma_z=100.0,      # very non-informative priors for the orbital elements
            sigma_vx=1.0,     # means based on o_prior
            sigma_vy=1.0,     # means based on o_prior
            sigma_vz=1.0,     # means based on o_prior            
            sigma_am=0.77,      # log10(A/m) stdev, default based on all master catalog objects
            am_mean=-0.5,       # log10(A/m) mean, default based on all master catalog objects
            N_samples=100000,
            use_prior=True,
            skip_fmin=True,
            mcmc=False,
            thinning=1,
            odir="./test_tracklets",
            proposal_cov=[0,0,0,0,0,0,0],  # covariance matrix of proposal step
            plot=True):

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
    print("range residuals (m)")
    for i in range(n_tracklets):
        print(1e3*(s_ranges[i]-m_range[i]))
    print("range_rate residuals (m/s)")
    for i in range(n_tracklets):    
        print(1e3*(s_range_rates[i]-m_range_rate[i]))

    o=o_prior.copy()
        
    A_to_m=o_prior.A/o_prior.m
    o.A=A_to_m
    o.m=1.0

    # don't start with real values to make map estimate realistic
    x0=n.array([o_prior.x+n.random.randn(1)*0.001,
                o_prior.y+n.random.randn(1)*0.001,
                o_prior.z+n.random.randn(1)*0.001,
                o_prior.vx+n.random.randn(1)*0.00001,
                o_prior.vy+n.random.randn(1)*0.00001,
                o_prior.vz+n.random.randn(1)*0.00001,
                n.log10(A_to_m)+n.random.randn(1)*0.0001])
    xtrue=n.array([o_prior.x,
                   o_prior.y,
                   o_prior.z,
                   o_prior.vx,
                   o_prior.vy,
                   o_prior.vz,
                   n.log10(A_to_m)])
    
    # object used for least squares c
#    o=spo.SpaceObject(a=a, e=e, i=i, raan=raan, aop=aop, mu0=mu0, A=A_to_m, m=1.0, mjd0=o_prior.mjd0)
    
    def ss(x):
        A_to_m=10**(x[6])

        if x[6] < -4.0 or x[6] > 100.0:
            return(1e6)

        o.update(x=x[0],
                 y=x[1],
                 z=x[2],
                 vx=x[3],
                 vy=x[4],
                 vz=x[5])
        o.A = A_to_m
        
        # forward model
        s_range, s_range_rate=sim_meas(o,m_time,rx_locs,tx_locs)

        s=0.0
        for i in range(n_tracklets):
            s+=0.5*(n.sum( ((m_range[i]-s_range[i])**2.0)/(m_range_std[i]**2.0))+ n.sum( ((m_range_rate[i]-s_range_rate[i])**2.0)/(m_range_rate_std[i]**2.0)))

            
        if use_prior:
            prior=0.0
            # Gaussian prior for log10(A/m) mean = -0.5, std=0.77, based on the
            # approximate distribution of log10(A/m) in the MASTER 2009 catalog
            prior += 0.5*(x[6] - am_mean)**2.0/sigma_am**2.0
            
            # Gaussian prior distribution for Keplerian parameters
            prior += 0.5*(x[0] - x0[0])**2.0/sigma_x**2.0 
            prior += 0.5*(x[1] - x0[1])**2.0/sigma_y**2.0
            prior += 0.5*(x[2] - x0[2])**2.0/sigma_z**2.0
            prior += 0.5*(x[3] - x0[3])**2.0/sigma_vx**2.0
            prior += 0.5*(x[4] - x0[4])**2.0/sigma_vy**2.0
            prior += 0.5*(x[5] - x0[5])**2.0/sigma_vz**2.0
#            print(prior)
#            print(s)
            s=s+prior
        return(s)

    print("maximum a posteriori search...")

    if skip_fmin:
        xhat=xtrue
    else:
        xhat=so.minimize(ss, x0, method=method).x
    
    print("xhat - xtrue")
    print(xhat)
    print("xtrue")
    print(xtrue)
    
    print("xhat - xtrue")
    print(xhat-xtrue)
    print("rank %d"%(rank))

    if rank == 0:
        ho=h5py.File("%s/true_state.h5"%(odir),"w")
        ho["state"]=xtrue
        ho["lin_cov"]=proposal_cov
        ho.close()
            
    o_fit=o.copy()
    o_fit.update(x=xhat[0],y=xhat[1],z=xhat[2],vx=xhat[3],vy=xhat[4],vz=xhat[5])
    o_fit.A=10**xhat[6]
    
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
        
#    comm.Barrier()
        
    if not mcmc:
        s_time, range_resid, vel_resid = compare_states(o_prior,o_fit,m_time,plot=plot)
        return(o_fit, s_time, range_resid, vel_resid, m_time)

    print("mcmc sampling... rank %d"%(rank))    
    # Single Component Adaptive Metropolis-hastings MCMC (SCAM) 
    chain=n.zeros([N_samples, len(x0)])
    xnow=n.copy(xhat)
    logss=-ss(xnow)   # log likelihood
    n_par=len(xhat)

#    if proposal_cov[0] == 0:
 #       print("we need a proposal distribution")
  #      exit(0)
    
    accept=0
    tries=0
    step=0.1
    prop_mean=n.zeros(proposal_cov.shape[0])
    for i in range(N_samples*thinning):
        xtry=n.copy(xnow)

        xtry=xtry + n.random.multivariate_normal(prop_mean,step*proposal_cov) 
        logss_try=-ss(xtry)  # log likelihood

        log_u = n.log(n.random.rand(1))
#        u = n.random.rand(1)[0]
        log_alpha = logss_try - logss

 #       alpha=n.exp(log_alpha)
        # proposal step
        # accept
        if log_u <= log_alpha:
#        if u <= alpha:            
            logss=logss_try
            xnow=xtry
            accept+=1
        #
        # TBD: Use DRAM to speed up sampling, because parameters
        # TBD2: use linearized error estimate to come up with cov matrix for proposal
        # are highly correlated.
        # After sufficiently many samples drawn with SCAM
        # estimate covariance with C=n.cov(n.transpose(chain))
        # use n.random.multivariate_normal to sample from C
        #
        chain[int(i/thinning),:]=xnow
        tries+=1.0

        # adaptive step scam. adjust proposal
        # based on acceptance probability
        if i%100 == 0 and i>0:
            print("step %d/%d"%(i,N_samples*thinning))
            print("logss %1.2f current state"%(logss))
            print(xnow)
            print("acceptance probability")            

            ratio=float(accept)/float(tries)
            print(ratio)

            if ratio > 0.7:
                step=step*1.5
            elif ratio < 0.3:
                step=step*0.5
                
            
            accept=0.0
            tries=0.0
        
        if i%100 == 0 and i>0:
            print("saving")
            ho=h5py.File("%s/chain-%03d.h5"%(odir,rank),"w")
            ho["chain"]=chain[0:(int(i/thinning)),:]
            ho.close()
    
    return(o_fit)

def plot_chain(dname):

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
    
    C_est=n.cov(n.transpose(c))
    
    hi=h5py.File("%s/true_state.h5"%(dname),"r")
    true_state=n.copy(hi["state"].value)
    lin_cov=n.copy(hi["lin_cov"].value)
    hi.close()

    print(n.sqrt(n.diag(C_est/lin_cov)))
    print(n.diag(C_est/lin_cov))
    plt.subplot(121)
    
    plt.pcolormesh(C_est,vmin=n.min(lin_cov),vmax=n.max(lin_cov))
    plt.title("MCMC covariance")
    plt.xlabel("Orbit parameter")
    plt.ylabel("Orbit parameter")    
    
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(lin_cov,vmin=n.min(lin_cov),vmax=n.max(lin_cov))
    plt.title("Linearized error covariance")
    plt.xlabel("Orbit parameter")
    plt.ylabel("Orbit parameter")    
    
    plt.colorbar()
    plt.show()
    print(n.sqrt(n.diag(C_est)))
    print(n.sqrt(n.diag(lin_cov)))

    print(c.shape)
    plt.subplot(231)
    plt.plot(c[:,0])
    plt.axhline(true_state[0])
    plt.subplot(232)
    plt.plot(c[:,1])
    plt.axhline(true_state[1])    
    plt.subplot(233)
    plt.plot(c[:,2])
    plt.axhline(true_state[2])        
    plt.subplot(234)
    plt.plot(c[:,3])
    plt.axhline(true_state[3])            
    plt.subplot(235)
    plt.plot(c[:,4])
    plt.axhline(true_state[4])                
    plt.subplot(236)
    plt.plot(c[:,5])
    plt.axhline(true_state[5])                
    plt.show()
    plt.plot(c[:,6])
    plt.axhline(true_state[6])                
    plt.show()

    
    d={"x":c[:,0],
       "y":c[:,1],
       "z":c[:,2],
       "vx":c[:,3],
       "vy":c[:,4],
       "vz":c[:,5],
       "log10(A/m)":c[:,6]
    }
    true_vals={"x":true_state[0],
               "y":true_state[1],
               "z":true_state[2],
               "vx":true_state[3],
               "vy":true_state[4],
               "vz":true_state[5],
               "log10(A/m)":true_state[6]}


    for i in range(0,3):
        print("par %d: mcmc map %f mcmc std %f, true %f lin std %f map resid %f (m)\n"%(i,1e3*n.mean(c[:,i]),1e3*n.std(c[:,i]),1e3*true_state[i],1e3*n.sqrt(lin_cov[i,i]),1e3*(true_state[i]-n.mean(c[:,i]))))
    for i in range(3,6):
        print("par %d: mcmc map %f mcmc std %f, true %f lin std %f map resid %f (m/s)\n"%(i,1e3*n.mean(c[:,i]),1e3*n.std(c[:,i]),1e3*true_state[i],1e3*n.sqrt(lin_cov[i,i]),1e3*(true_state[i]-n.mean(c[:,i]))))
    for i in range(6,7):
        print("par %d: mcmc map %f mcmc std %f , true %f lin std %f (m) map resid %f log10(A/m) \n"%(i,n.mean(c[:,i]),n.std(c[:,i]),true_state[i],n.sqrt(lin_cov[i,i]),(true_state[i]-n.mean(c[:,i]))))

    
    df = pd.DataFrame(d)

    pairs=[["x","y"],
           ["x","z"],
           ["y","z"],
           ["vx","vy"],
           ["vx","vz"],
           ["vy","vz"],
           ["x","log10(A/m)"],
           ["y","log10(A/m)"],
           ["x","log10(A/m)"],                      
           ["vx","log10(A/m)"],
           ["vy","log10(A/m)"],
           ["vx","log10(A/m)"]]

    if False:
        for p in pairs:
            #        g = sb.jointplot(x=p[0], y=p[1], data=df, kind="hex")

            plt.hist2d(df[p[0]],df[p[1]],bins=50)
            plt.xlabel(p[0])
            plt.ylabel(p[1])        
            plt.axvline(true_vals[p[0]])
            plt.axhline(true_vals[p[1]])        
            plt.show()

    grid = sb.PairGrid(data= df)

    def pairgrid_heatmap(x, y, **kws):
        rubbish=kws.pop("color")
        plt.hist2d(x, y, cmap="gist_yarg", **kws)

        
    def pairgrid_scatter(x, y, **kws):
        rubbish=kws.pop("color")
#        print(kws.pop("label"))        
#        cmap = sb.light_palette(kws.pop("color"), as_cmap=True)
        plt.plot(x, y, ".", color="black",alpha=0.01)
        
    def pairgrid_hist(x, **kws):
        rubbish=kws.pop("color")

 #       print(kws.pop("label"))
  #      print(x)
#        cmap = sb.light_palette(kws.pop("color"), as_cmap=True)
        plt.hist(x, color="black", **kws)

    
    # Map a histogram to the diagonal
    grid = grid.map_diag(pairgrid_hist, bins=50)
    # Map a density plot to the lower triangle
    grid = grid.map_lower(pairgrid_heatmap, bins=50)
#    grid = grid.map_upper(pairgrid_scatter)    
    #    sb.pairplot(df)
    #   grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
    plt.show()

    
    

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
    all_t_obs=n.array([])
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
            all_t_obs=n.concatenate([all_t_obs,t_obs])



    all_t_obs=n.unique(all_t_obs)
#    print(all_t_obs)    
 #   print(lin_t_obs)
            
    if rank==0:
        meas, fnames, ecef_stdevs = st.create_tracklet(o, e3d, all_t_obs, hdf5_out=True, ccsds_out=True, dname=dname, ignore_elevation_thresh=True, noise=True)
            
    error_cov,range_0=ec.linearized_errors(o,e3d,all_t_obs,plot=True,debug_output=True,include_drag=True)

    # then we read these measurements
    print("rank %d waiting"%(rank))
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
            range_stds.append(n.copy(h["m_range_std"].value))
            range_rate_stds.append(h["m_range_rate_std"].value)
            h.close()
          
    # get best fit space object    
    return(mcmc_od(all_t_meas, all_r_meas, all_rr_meas, range_stds, range_rate_stds, tx_locs, rx_locs, o, mcmc=mcmc, odir=dname, N_samples=N_samples, plot=plot,proposal_cov=error_cov))

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
    # oid 8400199
    # 008400199  0.1000E+01  0.6021E-01  0.5581E-01  35.180       7355.7  0.0008   65.82    26.92   304.76   295.20
    o=spo.SpaceObject(a=7355.7, e=0.0008, i=80.82, raan=26.92, aop=304.76, mu0=295.20, A=1/35.180, m=1.0, d=1.0)
    
    wls_state_est(mcmc=True, o=o, n_tracklets=4, track_length=1000.0, n_points=10, N_samples=500000)
#    wls_state_est(mcmc=False, oidx=8, n_tracklets=16, track_length=600.0, n_points=10, N_samples=50000)    
#    initial_iod_accuracy()
