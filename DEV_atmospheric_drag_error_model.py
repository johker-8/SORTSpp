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
def error_var(tau,t1=8943.65733709582,offset=3.31836111241377,alpha=3.02796924317894):
    print(tau)
    print(t1)
    print(alpha)
    print(offset)
    print((n.log(tau)-n.log(t1))*alpha + offset)
    return(n.exp((n.log(tau)-n.log(t1))*alpha + offset))

def atmospheric_errors_rw(o, n_tracklets=1, n_points=1, track_length=600.0, error0=40.0):

    e3d = rlib.eiscat_3d(beam='gauss')
    t = n.linspace(-2*24*3600, 2*24*3600, num=1000)
    t_all = n.linspace(0, 2*24*3600, num=100000)    
    passes, _, _, _, _ = st.find_pass_interval(t_all, o, e3d)
    print(len(passes[0]))
    if passes == None:
        return
    n_tracklets=n.min([n_tracklets,len(passes[0])])
    tracklet_idx = n.arange(n_tracklets)
    t_obs=[]
    # randomize tracklet positions
    n.random.shuffle(tracklet_idx)    
    for pi in tracklet_idx:
        p = passes[0][pi]
        mean_t=0.5*(p[1]+p[0])
        t_obs.append(mean_t)

    midx=[]
    for to in t_obs:
        midx.append(n.argmin(n.abs(t-to)))

    N_p=len(t)
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
#            print(dt)
            print(error_var(n.abs(dt)))
            S[row_idx]=error_var(n.abs(dt))
            row_idx+=1
    for i in range(n_meas):
        A[i,:]=A[i,:]/n.sqrt(S[i])
    Sp=n.linalg.inv(n.dot(n.transpose(A),A))
    plt.semilogy(t/3600.0,n.sqrt(n.diag(Sp)),label="Position error stdev")
    plt.ylabel("Position error standard deviation (m)")
    for i in range(n_tracklets):
        plt.axvline(t_obs[i]/3600.0,color="red",label="measurement")
    plt.xlabel("Time (hours)")
    plt.legend()
    plt.title("Atmospheric drag uncertainty related errors")
    plt.show()


    
o=spo.SpaceObject(a=7070.0, e=1e-4, i=89.0, raan=12.0, aop=23.0, mu0=32.0, A=10**(-2.0), m=1.0, d=1.0)
atmospheric_errors_rw(o,n_tracklets=3)


