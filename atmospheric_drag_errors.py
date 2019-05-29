#!/usr/bin/env python

'''
   Monte-Carlo sampling of errors due to atmospheric drag force uncertainty.

   Estimate a power-law model of error standard deviation in along-track direction (largest error).

   Juha Vierinen
'''
import time
import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#
# SORTS imports
#
import radar_library as rlib     # radar network definition
import population_library as plib
import simulate_tracking as st # find times of possible observations
import simulate_tracklet as s  # simulate a measurement
import space_object as so      # space object
import plothelp

from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


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

def atmospheric_errors(o,a_err_std=0.01,N_samps=100,plot=False,threshold_error=100.0, res = 500):
    """
    Estimate position errors as a function of time, assuming
    a certain error in atmospheric drag.
    """
 #   o0=so.SpaceObject(a=o.a,e=o.e,i=o.i,raan=o.raan,aop=o.aop,mu0=o.mu0,d=o.diam,A=o.A,m=o.m,C_D=o.C_D)
#    print(o0)


    t=10**(n.linspace(2,6.2,num=100))
    t_dt=n.copy(t)+1.0     
    ecef=o.get_orbit(t)

    print("n_days %d"%(n.max(t)/24.0/3600.0))
    C_D0=o.C_D
    err=n.copy(ecef)
    err[:,:]=0.0

    t0 = time.time()
    
    for i in range(N_samps):
        o1=o.copy()
        o1.mu0=n.random.rand(1)*360.0

        ecef=o1.get_orbit(t)
        ecef_dt=o1.get_orbit(t_dt)                
        at,norm,ct=get_inertial_basis(ecef,ecef_dt)
        
        C_D=C_D0 + C_D0*n.random.randn(1)[0]*a_err_std
        o1.C_D=C_D
        
        ecef1=o1.get_orbit(t)
        
        err_now=(ecef1-ecef)
        err[0,:]+=n.abs(err_now[0,:]*at[0,:]+err_now[1,:]*at[1,:]+err_now[2,:]*at[2,:])**2.0

        # difference in radius is the best estimate for radial distance error.
        err[1,:]+=n.abs(n.sqrt(ecef[0,:]**2.0+ecef[1,:]**2.0+ecef[2,:]**2.0) - n.sqrt(ecef1[0,:]**2.0+ecef1[1,:]**2.0+ecef1[2,:]**2.0))
#       and not this:  err[1,:]+=n.abs(err_now[0,:]*norm[0,:]+err_now[1,:]*norm[1,:]+err_now[2,:]*norm[2,:])**2.0        
        err[2,:]+=n.abs(err_now[0,:]*ct[0,:]+err_now[1,:]*ct[1,:]+err_now[2,:]*ct[2,:])**2.0

        elps = time.time() - t0
        print('{}/{} done - time elapsed {:<5.2f} h | estimated time remaining {:<5.2f}'.format(
            i+1,
            N_samps,
            elps/3600.0,
            elps/float(i+1)*float(N_samps - i - 1)/3600.0,
        ))

    ate=n.sqrt(err[0,:]/N_samps)
    if n.max(ate) > threshold_error:
        idx0=n.where(ate > threshold_error)[0][0]
        days=t/24.0/3600.0
        hour0=24.0*days[idx0]
    else:
        hour0=n.max(t/3600.0)
#    print(t)
    alpha=(n.log(err[0,-1]/N_samps)-n.log(err[0,46]/N_samps))/(n.log(t[-1])-n.log(t[46]))
#     (n.log(t[-1])-n.log(t[0]))*alpha=n.log(err[0,-1]/N_samps) 

    offset=n.log(err[0,46]/N_samps)
    t1 = t[46]
    var=n.exp((n.log(t)-n.log(t[46]))*alpha + offset)
    
    if plot:
        plt.loglog(t/24.0/3600.0,n.sqrt(err[0,:]/N_samps),label="Along track")
        plt.loglog(t/24.0/3600.0,n.sqrt(var),label="Fit",alpha=0.5)
        plt.loglog(t/24.0/3600.0,n.sqrt(err[1,:]/N_samps),label="Radial")
        plt.loglog(t/24.0/3600.0,n.sqrt(err[2,:]/N_samps),label="Cross-track")
        if n.max(ate) > threshold_error:    
            plt.axvline(days[idx0])
            plt.text(days[idx0],threshold_error,"$\\tau=%1.1f$ hours"%(24*days[idx0]))        
        plt.grid()
        plt.axvline(n.max(t)/24.0/3600.0)
        plt.xlim([0,n.max(t)/24.0/3600.0])
        plt.legend()
        plt.ylabel("Cartesian position error (m)")
        plt.xlabel("Time (days)")
#    plt.title("Atmospheric drag uncertainty related errors"%(alpha))
        plt.title("a %1.0f (km) e %1.2f i %1.0f (deg) aop %1.0f (deg) raan %1.0f (deg)\nA %1.2f$\pm$ %d%% (m$^2$) mass %1.2f (kg)\n$\\alpha=%1.1f$ $t_1=%1.1f$ $\\beta=%1.1f$"%(o.a,o.e,o.i,o.aop,o.raan,o.A,int(a_err_std*100.0),o.m,alpha,t1,offset))
#        plt.show()
    return(hour0,offset,t1,alpha)

    
if __name__ == "__main__":
    
    # Here's what I've gots to do. With MPI, because this will take a lot of time
    # For each detectable object:
    # - calculate number of passes over 24 hours
    # - calculate mean time between passes over 24 hours (24.0/N_passes)
    # - estimate amount of time that it takes to decorrelate
    # - save info n_passes_per_day, max_spacing, mean_spacing, is_maintainable
    # plot results for population
    
    radar_e3d = rlib.eiscat_3d(beam='interp', stage=1)

    from propagator_orekit import PropagatorOrekit
    from propagator_neptune import PropagatorNeptune

    base = '/home/danielk/IRF/E3D_PA/FP_sims/'

    # space object population
    pop_e3d = plib.filtered_master_catalog_factor(
        radar = radar_e3d,
        detectability_file = base + 'celn_20090501_00_EISCAT_3D_PropagatorSGP4_10SNRdB.h5',
        treshhold = 0.01,
        min_inc = 50,
        prop_time = 48.0,
        #propagator = PropagatorNeptune,
        #propagator_options = {
        #},
        propagator = PropagatorOrekit,
        propagator_options = {
            'in_frame': 'EME',
            'out_frame': 'ITRF',
        },
    )
    oids=[]
    taus=[]
    offsets=[]
    t1s=[]
    alphas=[]
    n_passes=[]
    mean_time_between_passes=[]                        
    t=n.linspace(0,24*3600,num=10000)

    # objs= list(pop_e3d.object_generator())
    objs=[]
    for o in pop_e3d.object_generator():
        objs.append(o)
        break
    for oi in range(comm.rank,len(objs),comm.size):
        o=objs[oi]
        passes,_,_,_,_=st.find_pass_interval(t, o, radar_e3d)
        
        if passes[0] == None:
            n_pass=0
        else:
            n_pass=len(passes[0])
            
        tau,offset,t1,alpha=atmospheric_errors(o,a_err_std=0.05,plot=True)
        print("%d %d/%d oid %d number of passes %d mean_time %1.1f (h) tau %1.1f (h)"%(comm.rank, oi,len(objs),o.oid,n_pass,24.0/(n_pass+1e-6),tau))
        plt.tight_layout()
        plt.show()
        #plt.savefig("report_plots/atmospheric_error_sample_%06d.png"%(oi))
        #plt.close()
        #plt.clf()

        oids.append(o.oid)
        n_passes.append(n_pass)
        t1s.append(t1)
        alphas.append(alpha)
        offsets.append(offset)
        mean_time_between_passes.append(24.0/(n_pass+1e-6))
        taus.append(tau)
        
        if oi%100 == 0 and oi > 0:
            ho=h5py.File("master/drag-%03d.h5"%(comm.rank),"w")
            ho["oids"]=oids
            ho["n_passes"]=n_passes
            ho["t1s"]=t1s
            ho["alphas"]=alphas
            ho["offsets"]=offsets
            ho["mean_time_between_passes"]=mean_time_between_passes
            ho["taus"]=taus
            ho.close()

        
    ho=h5py.File("master/drag-%03d.h5"%(comm.rank),"w")
    ho["oids"]=oids
    ho["n_passes"]=n_passes
    ho["t1s"]=t1s
    ho["alphas"]=alphas
    ho["offsets"]=offsets
    ho["mean_time_between_passes"]=mean_time_between_passes
    ho["taus"]=taus
    ho.close()
        
