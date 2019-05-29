#!/usr/bin/env python
#
#
import numpy as n
import matplotlib.pyplot as plt
from mpi4py import MPI

# SORTS imports
import population 
import simulate_scan as ss 
import simulate_tracklet as st
import radar_config as r

m=master.master_catalog()

t_obs=n.arange(0,24*60*4)*15

comm = MPI.COMM_WORLD    
for oid in range(comm.rank,len(m),comm.size):
    o=m.get_object(oid)
    #  try:
    print("object %d id=%d diam=%1.2g a=%1.2f e=%1.2f i=%1.2f raan=%1.2f aop=%1.2f mu0=%1.2f mass=%1.2g A=%1.2g factor=%1.2f"% \
    	(oid,o.oid,o.diam,o.a,o.e,o.i,o.raan,o.aop,o.mu0,o.m,o.A,o.factor))

    if o.a < 10000.0 and o.i > 40.0:    
        meas=st.create_tracklet(o,r,t_obs,dt=0.01,hdf5_out=True,ccsds_out=True,dname="./repeat_tracklets")
