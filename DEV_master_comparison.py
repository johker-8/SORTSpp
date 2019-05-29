#!/usr/bin/env python

import numpy as n
import scipy.constants as c
import matplotlib.pyplot as plt
import time
import os
import h5py

# SORTS imports
import radar_library as rl
import space_object as so
import simulate_scan as sc

from mpi4py import MPI
comm = MPI.COMM_WORLD

def sample_detections(radar,outdir="./master_validation_data",N=1000,a=7000,inc=70.0,diam=30.0,dt=24*3600):
#    if os.path.exists("%s/sample_a_%1.1f_i_%1.2f.h5"%(outdir,a,inc)):
 #       return
    tx_gains=[]
    rx_gains=[]
    ranges=[]
    range_rates=[]
    times=[]
    snrs=[]            
    
    # get all detections for one object during 24 hourscan
    for i in range(N):
        t0 = time.time()
        mu0=n.random.rand(1)*360.0
        o=so.space_object(a=a,e=0.0,i=inc,raan=0,aop=0,mu0=mu0,C_D=2.3,A=1.0,m=1.0,diam=diam)
        detections=sc.get_iods(o,radar,0,dt,eval_gain=True)
        det=detections[0]
        print(n.array(det["range_rate"])/2.0)
        n_det=len(det["tx_gain"])
        for di in range(n_det):
            tx_gains.append(det["tx_gain"][di])
            rx_gains.append(det["rx_gain"][di])
            snrs.append(det["snr"][di])
            ranges.append(det["range"][di])
            range_rates.append(det["range_rate"][di])
            times.append(det["tm"][di])                                                            

        t1=time.time()
        print("n_det %d wall clock time %1.2f"%(n_det,t1-t0))
    ho=h5py.File("%s/sample_a_%1.1f_i_%1.2f.h5"%(outdir,a,inc),"w")
    ho["a"]=a
    ho["inc"]=inc
    ho["tx_gains"]=tx_gains
    ho["rx_gains"]=rx_gains
    ho["ranges"]=ranges
    ho["range_rates"]=range_rates
    ho["times"]=times
    ho["snrs"]=snrs
    ho["dt"]=dt
    ho["N"]=N
    ho.close()

def sim_uhf(N_samples=10):
    outdir="./master_validation_data_uhf"
    os.system("mkdir -p %s"%(outdir))

    incs=n.arange(69,113,2)
    apogs=n.arange(6971,8371,100)
    print(len(apogs)*len(incs))
    radar=rl.eiscat_uhf()
    for inci,i in enumerate(incs):
        for ai in range(comm.rank,len(apogs),comm.size):
            a=apogs[ai]
            print("inc %f apog %f"%(i,a))
            sample_detections(radar,N=N_samples,a=a,inc=i,outdir=outdir)
            
def sim_esr(N_samples=10):
    outdir="./master_validation_data_esr"
    os.system("mkdir -p %s"%(outdir))

    incs=n.arange(79,104,1)
    apogs=n.arange(7371,8371,100)
    print(len(apogs)*len(incs))
    radar=rl.eiscat_svalbard()
    for inci,i in enumerate(incs):
        for ai in range(comm.rank,len(apogs),comm.size):
            a=apogs[ai]
            print("inc %f apog %f"%(i,a))
            sample_detections(radar,N=N_samples,a=a,inc=i,outdir=outdir)

    
if __name__ == "__main__":
    sim_uhf(N_samples=10)

