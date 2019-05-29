#!/usr/bin/env python

import time
import sys
import os

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plothelp
import dpt_tools as dpt
import radar_library as rlib
import ccsds_write

from propagator_neptune import PropagatorNeptune
from propagator_sgp4 import PropagatorSGP4
from propagator_orekit import PropagatorOrekit


radar = rlib.eiscat_3d()

p_sgp4 = PropagatorSGP4(
    polar_motion = False,
    out_frame = 'ITRF',
)

p_nept = PropagatorNeptune()
p_orekit = PropagatorOrekit(
    in_frame='TEME',
    out_frame='ITRF',
)

props = [
    p_sgp4,
    p_nept,
    p_orekit,
]

prop_dat = [
    ('-b','SGP4'),
    ('-g','NEPTUNE'),
    ('-k','Orekit'),
]

ut0 = 1241136000.0

# modified Julian day epoch
mjd0 = dpt.jd_to_mjd(dpt.unix_to_jd(ut0))
R_e = 6371.0
m_to_A = 128.651
mass = 0.8111E+04

init_data = {
    'a': 7159.5*1e3,
    'e': 0.0001,
    'inc': 98.55,
    'raan': 248.99,
    'aop': 90.72,
    'mu0': 47.37,
    'mjd0': mjd0,
    'C_D': 2.3,
    'C_R': 1.0,
    'm': mass,
    'A': mass/m_to_A,
}


def try_OD_ECI_to_ECEF():
    mjd0 = 54952.051758969
    state = np.array([-2518.55722577, -4474.66843264, 4997.78582595, 1.14790783, 5.19632151, 5.22053125])*1e3 
    x, y, z, vx, vy, vz = state

    ecef_ref = np.array([5130.2875669596606, 265.49822165794365, 4995.3287514630274, -4.9810366392179519, -2.1915354990789666, 5.2217201835482943])*1e3

    t = np.array([60.0])

    for prop, dat in zip(props, prop_dat):
        ecefs = prop.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0, 
            m = init_data['m'], 
            C_D = init_data['C_D'],
            C_R = init_data['C_R'],
            A = init_data['A'],

        )

        print('\n' + '='*10 + ' COMPARE {} '.format(dat[1]) + '='*10)
        print(('{:<12}: '.format('ECI initial') + '{:<15.6f} |'*6).format(*state.tolist()))
        print(('{:<12}: '.format(dat[1]) + '{:<15.6f} |'*6).format(*ecefs[:,0].tolist()))
        print(('{:<12}: '.format('Reference') + '{:<15.6f} |'*6).format(*ecef_ref.tolist()))
        print(('{:<12}: '.format('Difference') + ('{:<15.6f} | '*6).format(*(ecefs[:,0] - ecef_ref).tolist())))


def try_compare():

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)

    fig2 = plt.figure(figsize=(15,15))
    axs = [
        fig2.add_subplot(311),
        fig2.add_subplot(312),
        fig2.add_subplot(313),
    ]

    fig3 = plt.figure(figsize=(15,15))
    ax_compr = fig3.add_subplot(211)
    ax_compv = fig3.add_subplot(212)

    ecef_tx = radar._tx[0].ecef

    ax.plot([ecef_tx[0]], [ecef_tx[1]], [ecef_tx[2]], 'or', label='EISCAT 3D')

    t = np.arange(0, 3*3600.0, 60.0, dtype=np.float)

    for prop, dat in zip(props, prop_dat):
        ecefs = prop.get_orbit(t, **init_data)

        if dat[1] == 'SGP4':
            _ecef = ecefs

        print('='*30)
        print('Propagating with: {}\n'.format(prop))
        ax.plot(ecefs[0,:], ecefs[1,:], ecefs[2,:],dat[0], label = dat[1], alpha = 0.75)

        for ind in range(3):
            axs[ind].plot(t/3600.0, ecefs[ind,:]*1e-3, dat[0], label = dat[1], alpha = 0.4)

        dr = np.sqrt(np.sum((_ecef[:3,:] - ecefs[:3,:])**2, axis=0))
        dv = np.sqrt(np.sum((_ecef[3:,:] - ecefs[3:,:])**2, axis=0))

        ax_compr.plot(t/3600.0, dr*1e-3, dat[0], label = dat[1], alpha = 1)
        ax_compv.plot(t/3600.0, dv*1e-3, dat[0], label = dat[1], alpha = 1)

    ax_compr.set(
        xlabel='Time [h]',
        ylabel='$|\mathbf{r}_i - \mathbf{r}_{SGP4}|$ difference ITRF [km]',
    )
    ax_compv.set(
        xlabel='Time [h]',
        ylabel='$|\mathbf{v}_i - \mathbf{v}_{SGP4}|$ difference ITRF [km/s]',
    )
    ax_compr.set_title('ENVISAT Propagation comparison')
    ax_compr.legend()
    ax_compv.legend()

    axs[0].set(
        xlabel='Time [h]',
        ylabel='X position ITRF [km]',
    )
    axs[0].legend()
    axs[1].set(
        xlabel='Time [h]',
        ylabel='Y position ITRF [km]',
    )
    axs[2].set(
        xlabel='Time [h]',
        ylabel='Z position ITRF [km]',
    )
    axs[0].set_title('ENVISAT Propagation comparison')

    ax.set_title('ENVISAT Propagation comparison')
    ax.legend()



def test_envisat():
    import space_object as so

    mass=0.8111E+04
    diam=0.8960E+01
    m_to_A=128.651
    a=7159.5
    e=0.0001
    i=98.55
    raan=248.99
    aop=90.72
    M=47.37
    A=mass/m_to_A
    # epoch in unix seconds
    ut0=1241136000.0
    # modified julian day epoch
    mjd0 = dpt.jd_to_mjd(dpt.unix_to_jd(ut0))

    print(mjd0)
    o=so.SpaceObject(
        a=a,e=e,i=i,raan=raan,aop=aop,mu0=M,C_D=2.3,A=A,m=mass,diam=diam,mjd0=mjd0,
        #propagator = PropagatorNeptune,
        propagator = PropagatorSGP4,
        propagator_options = {
            'out_frame': 'ITRF'
        },
    )

    e3d = rlib.eiscat_3d()
    print("EISCAT Skibotn location x,y,z ECEF (meters)")
    print(e3d._tx[0].ecef)
    ski_ecef=e3d._tx[0].ecef

    print("EISCAT Skibotn location %1.3f %1.3f %1.3f (lat,lon,alt)"%(e3d._tx[0].lat,e3d._tx[0].lon,0.0))

    t_obs=np.linspace(4440,5280,num=100)+31.974890
    t_obs2=np.linspace(4440,5280,num=100)+31.974890 + 1.0

    print("MJD %1.10f %sZ"%(mjd0+t_obs[0]/3600.0/24.0,ccsds_write.unix2datestr(ut0+t_obs[0])))
    ecef=o.get_state(t_obs)
    ecef2=o.get_state(t_obs2)

    print("ECEF state x,y,z,vx,vy,vz  (km and km/s)")
    print(ecef[:,0]/1e3)

    print("Time (UTC)          Range (km)  Vel (km/s)  ECEF X (km) ECEF Y (km) ECEF Z (km)")
    for i in range(len(t_obs)):
        dist=np.linalg.norm(ecef[0:3,i]-ski_ecef)
        dist2=np.linalg.norm(ecef2[0:3,i]-ski_ecef)

        vel=(dist2-dist)/1.0
        print("%s   %1.3f %1.3f %1.3f %1.3f %1.3f"%(ccsds_write.unix2datestr(ut0+t_obs[i]),dist/1e3,vel/1e3,ecef[0,i]/1e3,ecef[1,i]/1e3,ecef[2,i]/1e3))


if __name__=='__main__':
    #try_compare()
    #try_OD_ECI_to_ECEF()
    test_envisat()

    plt.show()