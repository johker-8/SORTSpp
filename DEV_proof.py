#!/usr/bin/env python
#
# PROOF style simulation of a beampark measurement
#
import numpy as n
import scipy.constants as c
import matplotlib.pyplot as plt
import time
import os
import h5py
import glob

# SORTS imports
import radar_library as rl
import space_object as so
import simulate_scan as sc

from mpi4py import MPI
comm = MPI.COMM_WORLD

def read_master(fname="master/pop_20180201_00_1cm.pop"):
    a=n.genfromtxt(fname,skip_header=4)
    idx=n.argsort(a[:,3])
    idx=idx[::-1]
    m={"id":a[idx,0],"factor":a[idx,1],"mass":a[idx,2],"diam":a[idx,3],"m/A":a[idx,4],"a":a[idx,5],"e":a[idx,6],"i":a[idx,7],"raan":a[idx,8],"aop":a[idx,9],"m":a[idx,10]}
    return(m)
    
def proof(radar,m,outdir="./proof_uhf",max_a=9000.0):
    
    os.system("mkdir -p %s"%(outdir))

    min_inc = radar._tx[0].lat-2
    max_inc = 180.0-radar._tx[0].lat+2
    
    n_dets=0.0
    n_objs=len(m["id"])
    for oi in range(comm.rank,n_objs,comm.size):

        rfl=glob.glob("%s/det_%09d_*.h5"%(outdir,m["id"][oi]))
        if len(rfl)>0:
            print("skipping.")
            continue
        
        fact=m["factor"][oi]
        factor=0.0
        if fact < 1.0:
            if n.random.rand(1) < fact:
                factor=1.0
        else:
            factor=n.round(fact)
                
        for fi in range(int(factor)):
            print("oi %d/%d fi %d/%d n_dets %d a %f diam %f"%(oi,n_objs,fi,int(factor),n_dets,m["a"][oi],m["diam"][oi]))

            if (m["i"][oi] > min_inc) and (m["i"][oi] < max_inc) and (m["a"][oi] < max_a):
                mu0=n.random.rand(1)*360.0
                o=so.space_object(a=m["a"][oi],e=m["e"][oi],i=m["i"][oi],raan=m["raan"][oi],aop=m["aop"][oi],mu0=mu0,A=m["mass"][oi]/m["m/A"][oi],m=m["mass"][oi],diam=m["diam"][oi])
                d=sc.get_iods(o,radar,0,24.0*3600.0,eval_gain=True)[0]
                n_dets+=len(d["range"])
                if len(d["range"]) > 0:
                    for di in range(len(d["range"])):
                        print("det oi %d fi %d di %d angle %1.3f"%(oi,fi,di,d["on_axis_angle"][di]))
                        fname="%s/det_%09d_%06d_%03d.h5"%(outdir,m["id"][oi],fi,di)
                        print(fname)
                        ho=h5py.File(fname,"w")
                        ho["time"]=d["tm"][di]
                        ho["range"]=d["range"][di]
                        ho["range_rate"]=d["range_rate"][di]
                        ho["snr"]=d["snr"][di]
                        ho["oid"]=m["id"][oi]                    
                        ho["rx_gain"]=d["tx_gain"][di]
                        ho["tx_gain"]=d["tx_gain"][di]
                        ho["on_axis_angle"]=d["on_axis_angle"][di]                                                                                                                        
                        ho.close()
    comm.barrier()    

    
if __name__ == "__main__":
    m=read_master()
    uhf=True
    esr=False
    if uhf:
        radar=rl.eiscat_uhf()
        proof(radar,m,outdir="proof_uhf")
    if esr:
        radar=rl.eiscat_svalbard()
        proof(radar,m,outdir="proof_esr")        
