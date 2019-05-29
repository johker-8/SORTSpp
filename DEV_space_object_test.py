import numpy as n
import os

import sorts_config as sconf
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import plothelp
import space_object as so
import simulate_scan as scan
import time
import radar_library as rl


# test propagation
with so.space_object(a=7000,e=0.0,i=69,raan=0,aop=0,mu0=0,C_D=2.3,A=1.0,m=1.0) as o:
    t=n.linspace(0,24*3600,num=10000)
    ecefs=o.get_orbit(t)
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)
    plothelp.draw_earth(ax)
    plothelp.draw_radar(ax,69,19)    
    ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.5,color="black")
    plt.title("Orbital propagation test")
    #plt.savefig("orbitprop.png")
    plt.show()


with so.space_object(a=7000,e=0.0,i=72,raan=0,aop=0,mu0=0,C_D=2.3,A=1.0,m=1.0,diam=0.1) as o:
    t0 = time.time()
    radar=rl.eiscat_3d()
    # get all IODs for one object during 24 hourscan
    det_times=scan.get_iods(o,radar,0,24*3600)
    
    print(det_times)
    
    t1=time.time()
    print("wall clock time %1.2f"%(t1-t0))
    
