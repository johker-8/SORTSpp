#!/usr/bin/env python
#
#
# Linearized model with error behaviour that approximately follows anomalous diffusion
# 
import space_object as spo
import simulate_tracking as st
import radar_library as rlib
import numpy as n
import matplotlib.pyplot as plt

# anomalous diffusion (superdiffusion model)
# for atmospheric density related measurement errors
def error_var(tau,t1=8943.65733709582,beta=7.8,alpha=4.1):
    return(n.exp((n.log(tau)-n.log(t1))*alpha + beta))

def atmospheric_errors_rw(t_obs, t=n.linspace(-2*24*3600, 2*24*3600, num=1000), alpha=4.1, beta=7.8, t1=8943.65, error0=40.0,loglog=False, show_model=False,plot=False):
    """
    Use difference operators to specify a power law anomalous diffusion model for atmospheric measurement errors
    """    
    midx=[]
    for to in t_obs:
        midx.append(n.argmin(n.abs(t-to)))
    
    N_p=len(t)
    n_tracklets=len(t_obs)
    
    n_meas=n_tracklets+n_tracklets*(N_p-1)
    A=n.zeros([n_meas,N_p])

    row_idx=0
    idx=n.arange(N_p)
    S=n.zeros(n_meas)
    for i in range(n_tracklets):
        A[row_idx,midx[i]]=1.0
        S[row_idx]=error0**2.0
        row_idx+=1
        idx2=n.setdiff1d(idx,midx[i])
        for j in idx2:
            A[row_idx,midx[i]]=1.0
            A[row_idx,j]=-1.0
            dt=t[midx[i]]-t[j]
            S[row_idx]=error_var(n.abs(dt),alpha=alpha, beta=beta, t1=t1)
            row_idx+=1
    for i in range(n_meas):
        A[i,:]=A[i,:]/n.sqrt(S[i])
        
    # error covariance that takes into account all the correlations
    # provided by the anomalous diffusion model
    Sp=n.linalg.inv(n.dot(n.transpose(A),A))
    pos_err_std=n.sqrt(n.diag(Sp))

    if plot:
        if loglog:
            plt.loglog(t/3600.0,pos_err_std,label="Position error stdev")
            if show_model:
                plt.loglog(t/3600.0,n.sqrt(error_var(t,alpha=alpha,beta=beta,t1=t1)),label="Atmospheric error model")
        else:
            plt.semilogy(t/3600.0,pos_err_std,label="Position error stdev")
            if show_model:
                plt.semilogy(t/3600.0,n.sqrt(error_var(n.abs(t),alpha=alpha,beta=beta,t1=t1)),label="Atmospheric error model")
        
        plt.ylabel("Position error standard deviation (m)")
        for i in range(n_tracklets):
            plt.axvline(t_obs[i]/3600.0,color="red")
        plt.xlabel("Time (hours)")
        plt.ylim([10,2*n.max(pos_err_std)])

        plt.title("Atmospheric drag uncertainty related errors\nn_tracklets=%d"%(n_tracklets))
    return(pos_err_std)


def find_passes(o,radar,n_tracklets=1, max_t=4*24*3600.0):
    t_all = n.linspace(0, max_t, num=100000)    
    passes, _, _, _, _ = st.find_pass_interval(t_all, o, radar)

    if passes == None:
        return
    print(passes[0])
    print("n_tracklets %d"%(len(passes[0])))
    n_tracklets=n.min([n_tracklets,len(passes[0])])
    tracklet_idx = n.arange(n_tracklets)
    t_obs=[]
    # randomize tracklet positions
    n.random.shuffle(tracklet_idx)

    for pi in tracklet_idx:
        p = passes[0][pi]
        mean_t=0.5*(p[1]+p[0])
        t_obs.append(mean_t)
    return(t_obs)

def test_single():
    """
    Test plot to verify that the linearized random process follows the log-log power law
    of the anomalous diffusion process.
    """
    t=n.linspace(600.0, 10*24*3600, num=1000)
    t_obs=n.array([0.0])
    atmospheric_errors_rw(t_obs,t=t,loglog=True,show_model=True,plot=True)
    plt.savefig("report_plots/atmospheric_single.png")
    plt.close()
    plt.clf()

def test_multi():
    
    t_obs_all=n.array([0,96,48,24,72,12,36,60,84,6,18,30,42,54,66,78,90])*3600.0
    
    for i in range(len(t_obs_all)):
        t_obs=t_obs_all[0:(i+1)]
        print(t_obs)
        t=n.linspace(-3*24*3600, 7*24*3600, num=1000)
        atmospheric_errors_rw(t_obs,t=t,plot=True)
        plt.savefig("report_plots/atmospheric_drag_%02d.png"%(i+1))
        plt.close()
        plt.clf()

if __name__ == "__main__":
    test_single()    
    test_multi()
