#!/usr/bin/env python
#
# compare different propagators
# 
from keplerian_simple import simple_propagator as p_kep
from keplerian_sgp4 import sgp4_propagator as p_sgp4
from keplerian_opi import neptune_propagator as p_nep

import numpy as n
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.constants import au
from scipy.constants import G
M_sol = 1.98847e30

import time

#turn on TeX interperter
plt.rc('text', usetex=True)

plt_c = -1
plot_on = [1,1,1,1,1]
size_in = (15, 7)

SIM_ID = 4

mjd0=2457126.2729
C_D=2.3
m=8000
A=1.0
inc0=80.0
raan0=10
aop0=10.0
mu00=40.0

#
if SIM_ID == 1:
    a0=9000
    e0=0.1
    sim_time = [7]
elif SIM_ID == 2:
    a0=7000
    e0=0.0
    sim_time = [48,6]
elif SIM_ID == 3:
    a0=36000
    e0=0.0
    sim_time = [20]
    inc0=90
elif SIM_ID == 4:
    a0=36000
    e0=0.0
    sim_time = [20]
    inc0=0.001
elif SIM_ID == 5:
    a0=36000
    e0=0.7
    sim_time = [40]
    inc0=87.0

for tend in sim_time:
    plt_c = -1

    t=n.arange(0.0,tend*3600,30.0)
    td = t/(3600*24)

    t0=time.time()
    with p_kep() as p0:
        ecefs_simple=p0.get_orbit(t,a=a0,e=e0,inc=inc0,raan=raan0,aop=aop0,mu0=mu00,mjd0=mjd0,C_D=C_D,m=m,A=A)
        t1=time.time()
        print(1e6*(t1-t0)/10000.0)

    t0=time.time()
    with p_sgp4() as p1:
        ecefs_sgp4=p1.get_orbit(t,a=a0,e=e0,inc=inc0,raan=raan0,aop=aop0,mu0=mu00,mjd0=mjd0,C_D=C_D,m=m,A=A)
        t1=time.time()
        print(1e6*(t1-t0)/10000.0)

    t0=time.time()
    with p_nep() as p2:
        ecefs_opi=p2.get_orbit(t,a=a0,e=e0,inc=inc0,raan=raan0,aop=aop0,mu0=mu00,mjd0=mjd0,C_D=C_D,m=m,A=A)
        t1=time.time()
        print(1e6*(t1-t0)/10000.0)

    ecefs_opi = ecefs_opi[:3,:]
    ecef_diff_opi_sgp4 = (ecefs_sgp4 - ecefs_opi)**2
    ecef_diff_opi_sgp4 = n.sqrt(n.sum(ecef_diff_opi_sgp4,0))

    ecefs_sgp4_v = n.sqrt(n.diff(ecefs_sgp4[0,:])**2.0+n.diff(ecefs_sgp4[1,:])**2.0+n.diff(ecefs_sgp4[2,:])**2.0)/1e3
    ecefs_opi_v = n.sqrt(n.diff(ecefs_opi[0,:])**2.0+n.diff(ecefs_opi[1,:])**2.0+n.diff(ecefs_opi[2,:])**2.0)/1e3
    ecefs_simple_v = n.sqrt(n.diff(ecefs_simple[0,:])**2.0+n.diff(ecefs_simple[1,:])**2.0+n.diff(ecefs_simple[2,:])**2.0)/1e3

    plt_c += 1
    if plot_on[plt_c] == 1:
            
        fig = plt.figure(figsize=size_in,dpi=80)

        for i in range(3):
            ax = fig.add_subplot(3,1,i+1)
            ax.plot(td,ecefs_sgp4[i,:],label="sgp4")
            ax.plot(td,ecefs_opi[i,:],label="neptune")
            #ax.plot(td,ecefs_simple[i,:],label="simple")
            ax.legend()
            ax.set_title('Compare pos',fontsize=24)
            ax.set_xlabel('Days',fontsize=20)
            ax.set_ylabel('Axis %d'%(i),fontsize=20)
            plt.tight_layout()

    plt_c += 1
    if plot_on[plt_c] == 1:
        fig = plt.figure(figsize=size_in,dpi=80)

        ax = fig.add_subplot(1,1,1)
        ax.plot(td[1:],ecefs_sgp4_v,label="sgp4")
        ax.plot(td[1:],ecefs_opi_v,label="neptune")
        #ax.plot(td[1:],ecefs_simple_v,label="simple")
        ax.legend()
        ax.set_title('Compare vel',fontsize=24)
        ax.set_xlabel('Days',fontsize=20)
        ax.set_ylabel('X-pos',fontsize=20)

    plt_c += 1
    if plot_on[plt_c] == 1:

        fig = plt.figure(figsize=size_in,dpi=80)

        E_sgp4 = ecefs_sgp4_v**2*0.5*m - G*M_sol*m/n.sqrt(n.sum(ecefs_sgp4[:,1:]**2,axis=0))
        E_opi = ecefs_opi_v**2*0.5*m - G*M_sol*m/n.sqrt(n.sum(ecefs_opi[:,1:]**2,axis=0))
        E_simple = ecefs_simple_v**2*0.5*m - G*M_sol*m/n.sqrt(n.sum(ecefs_simple[:,1:]**2,axis=0))

        ax = fig.add_subplot(1,1,1)
        ax.plot(td[1:],E_sgp4/E_sgp4[0],label="sgp4")
        ax.plot(td[1:],E_opi/E_opi[0],label="neptune")
        #ax.plot(td[1:],E_simple/E_simple[0],label="simple")
        ax.legend()
        ax.set_title('Compare energy',fontsize=24)
        ax.set_xlabel('Days',fontsize=20)
        ax.set_ylabel('Relative E',fontsize=20)

        fig = plt.figure(figsize=size_in,dpi=80)

        ax = fig.add_subplot(1,1,1)
        ax.plot(td[1:],E_sgp4,label="sgp4")
        ax.plot(td[1:],E_opi,label="neptune")
        #ax.plot(td[1:],E_simple,label="simple")
        ax.legend()
        ax.set_title('Compare energy',fontsize=24)
        ax.set_xlabel('Days',fontsize=20)
        ax.set_ylabel('E',fontsize=20)

    plt_c += 1
    if plot_on[plt_c] == 1:

        fig = plt.figure(figsize=size_in,dpi=80)
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(ecefs_sgp4[0,:],ecefs_sgp4[1,:],ecefs_sgp4[2,:],label="sgp4")
        ax.plot(ecefs_opi[0,:],ecefs_opi[1,:],ecefs_opi[2,:],label="neptune")
        #ax.plot(ecefs_simple[0,:],ecefs_simple[1,:],ecefs_simple[2,:],label="simple")
    
    plt_c += 1
    if plot_on[plt_c] == 1:

        fig = plt.figure(figsize=size_in,dpi=80)
        ax = fig.add_subplot(1,1,1)
        ax.plot(td,ecef_diff_opi_sgp4*1e-3,label="sgp4-neptune")
        ax.legend()
        ax.set_title('Compare pos',fontsize=24)
        ax.set_xlabel('Days',fontsize=20)
        ax.set_ylabel('Orbit deviation [km]',fontsize=20)
        


plt.show()