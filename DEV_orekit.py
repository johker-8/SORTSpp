#!/usr/bin/env python

from mpi4py import MPI
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import plothelp

import propagator_orekit
from propagator_orekit import PropagatorOrekit
import dpt_tools as dpt
import orbit_verification as over
import TLE_tools as tle

R_e = 6371.0


def trySteps():
    p = PropagatorOrekit()

    line1 = '1 43947U 19006A   19069.62495353  .00222140  22344-4  39735-3 0  9995'
    line2 = '2 43947  96.6263 343.1609 0026772 226.5664 224.4731 16.01032328  7176'

    mjd0 = dpt.jd_to_mjd(tle.tle_jd(line1))

    state, epoch = tle.TLE_to_TEME(line1, line2)

    x, y, z, vx, vy, vz = state

    kwargs = dict(m=800., C_R=1., C_D=2.3, A=2.0)
    t_end = 24*3600.0
    t_vecs = [np.linspace(0,t_end, num=ind) for ind in [100,200,500,1000,10000]]

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)

    for t in t_vecs:
        pv = p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0, **kwargs)

        ax.plot(t/3600., np.linalg.norm(pv[:3,:]*1e-3, axis=0), label='{} steps @ {} s step size'.format(len(t),t_end/float(len(t))))

    ax.set(
        xlabel='Time [h]',
        ylabel='Range [km]',
        title='Step number difference',
    )
    plt.legend()
    plt.show()

def tryActivity():

    line1 = '1 43947U 19006A   19069.62495353  .00222140  22344-4  39735-3 0  9995'
    line2 = '2 43947  96.6263 343.1609 0026772 226.5664 224.4731 16.01032328  7176'

    mjd0 = dpt.jd_to_mjd(tle.tle_jd(line1))

    state, epoch = tle.TLE_to_TEME(line1, line2)

    x, y, z, vx, vy, vz = state

    p1 = PropagatorOrekit(
        solar_activity_strength = 'STRONG',
    )

    p2 = PropagatorOrekit(
        solar_activity_strength = 'WEAK',
    )

    kwargs = dict(m=800., C_R=1., C_D=2.3, A=4.0)

    t = np.linspace(0,3*24*3600.0, num=5000, dtype=np.float64)

    pv1 = p1.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0, **kwargs)
    pv2 = p2.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0, **kwargs)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(211)
    ax.plot(t/3600., np.linalg.norm(pv1[:3,:] - pv2[:3,:], axis=0)*1e-3)

    ax.set(
        xlabel='Time [h]',
        ylabel='Position difference [km]',
        title='Strong vs weak solar activity',
    )
    ax = fig.add_subplot(212)
    ax.plot(t/3600., np.linalg.norm(pv1[:3,:]*1e-3, axis=0), label='Weak')
    ax.plot(t/3600., np.linalg.norm(pv2[:3,:]*1e-3, axis=0), label='Strong')

    ax.set(
        xlabel='Time [h]',
        ylabel='Range [km]',
    )
    plt.legend()
    plt.show()



def try1():
    t0 = time.time()
    p = PropagatorOrekit()
    print('init time: {} sec'.format(time.time() - t0))

    init_data = {
        'a': 9000e3,
        'e': 0.1,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 8000,
        'A': 1.0,
    }
    t = np.linspace(0,24*3600.0, num=1000, dtype=np.float)

    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (first): {} sec'.format(time.time() - t0))


    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (second): {} sec'.format(time.time() - t0))


    t0 = time.time()
    init_data['C_D'] = 1.0
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (new param): {} sec'.format(time.time() - t0))

    times = []
    nums = []
    for num in range(100,1000,100):
        t = np.arange(0, 24*3600.0/1000.0*num, 24*3600.0/1000.0, dtype=np.float)
        t0 = time.time()
        ecefs = p.get_orbit(t, **init_data)
        times.append(time.time() - t0)
        nums.append(num)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(211)
    ax.plot(nums, times)
    ax = fig.add_subplot(212)
    ax.plot(nums, np.array(times)/np.array(nums, dtype=np.float))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ecefs[0,:], ecefs[1,:], ecefs[2,:],".",color="green")
    plt.show()

def try2():

    p = PropagatorOrekit()
    d_v = [0.1, 0.2]
    init_data = {
        'a': (R_e + 400.0)*1e3,
        'e': 0.01,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 3.0,
        'A': np.pi*(d_v[0]*0.5)**2,
    }
    t = np.linspace(0,3*3600.0, num=500, dtype=np.float)

    ecefs1 = p.get_orbit(t, **init_data)

    init_data['A'] = np.pi*(d_v[1]*0.5)**2
    ecefs2 = p.get_orbit(t, **init_data)

    dr = np.sqrt(np.sum((ecefs1[:3,:] - ecefs2[:3,:])**2, axis=0))
    dv = np.sqrt(np.sum((ecefs1[3:,:] - ecefs2[3:,:])**2, axis=0))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(211)
    ax.plot(t/3600.0, dr)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Position difference [m]')
    ax.set_title('Propagation difference diameter {} vs {} m'.format(d_v[0], d_v[1]))
    ax = fig.add_subplot(212)
    ax.plot(t/3600.0, dv)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Velocity difference [m/s]')


    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],".",color="green")
    ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],".",color="red")
    plt.show()


def tryMPI():

    comm = MPI.COMM_WORLD
    rank = comm.rank

    p = PropagatorOrekit()
    init_data = {
        'a': 9000e3,
        'e': 0.1,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 8000,
        'A': 1.0,
    }
    t = np.linspace(0,24*3600.0, num=1000, dtype=np.float)
    t0 = time.time()
    print('start rank {}'.format(rank))
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (rank {}): {} sec'.format(rank, time.time() - t0))

    comm.barrier()
    print('pass barrier {}'.format(rank))
    
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time, no init work, (rank {}): {} sec'.format(rank, time.time() - t0))


def trySpeed():
    init_data = {
        'a': 9000e3,
        'e': 0.1,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 8000,
        'A': 1.0,
    }
    t0 = time.time()
    p = PropagatorOrekit()
    print('-'*25 + '\n' + 'init time (full): {} sec'.format(time.time() - t0))
    t = np.linspace(0,24*3600.0, num=1000, dtype=np.float)
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (full): {} sec'.format(time.time() - t0))
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (full, second): {} sec'.format(time.time() - t0))

    t0 = time.time()
    p = PropagatorOrekit(earth_gravity='Newtonian')
    print('-'*25 + '\n' + 'init time (Newtonian): {} sec'.format(time.time() - t0))
    t = np.linspace(0,24*3600.0, num=1000, dtype=np.float)
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (Newtonian): {} sec'.format(time.time() - t0))
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (Newtonian, second): {} sec'.format(time.time() - t0))


    t0 = time.time()
    p = PropagatorOrekit(solarsystem_perturbers=[])
    print('-'*25 + '\n' + 'init time (no solarsystem_perturbers): {} sec'.format(time.time() - t0))
    t = np.linspace(0,24*3600.0, num=1000, dtype=np.float)
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (no solarsystem_perturbers): {} sec'.format(time.time() - t0))
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (no solarsystem_perturbers, second): {} sec'.format(time.time() - t0))

    t0 = time.time()
    p = PropagatorOrekit(drag_force=False)
    print('-'*25 + '\n' + 'init time (no drag_force): {} sec'.format(time.time() - t0))
    t = np.linspace(0,24*3600.0, num=1000, dtype=np.float)
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (no drag_force): {} sec'.format(time.time() - t0))
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (no drag_force, second): {} sec'.format(time.time() - t0))

    t0 = time.time()
    p = PropagatorOrekit(radiation_pressure=False)
    print('-'*25 + '\n' + 'init time (no radiation_pressure):: {} sec'.format(time.time() - t0))
    t = np.linspace(0,24*3600.0, num=1000, dtype=np.float)
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (no radiation_pressure): {} sec'.format(time.time() - t0))
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (no radiation_pressure, second): {} sec'.format(time.time() - t0))

    t0 = time.time()
    p = PropagatorOrekit(earth_gravity='Newtonian', radiation_pressure=False, solarsystem_perturbers=[])
    print('-'*25 + '\n' + 'init time (only drag + kep):: {} sec'.format(time.time() - t0))
    t = np.linspace(0,24*3600.0, num=1000, dtype=np.float)
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (only drag + kep): {} sec'.format(time.time() - t0))
    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (only drag + kep, second): {} sec'.format(time.time() - t0))


def tryFrame():

    p = PropagatorOrekit(in_frame='ITRF', out_frame='ITRF')

    print(p)

    init_data = {
        'a': (R_e + 400.0)*1e3,
        'e': 0.01,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 3.0,
        'A': np.pi*1.0**2,
    }
    t = np.linspace(0,3*3600.0, num=500, dtype=np.float)

    ecefs1 = p.get_orbit(t, **init_data)

    print(p)

    p = PropagatorOrekit(in_frame='EME', out_frame='ITRF')

    ecefs2 = p.get_orbit(t, **init_data)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],".",color="green",label='Initial frame: ITRF')
    ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],".",color="red",label='Initial frame: EME')
    plt.legend()
    plt.show()


def tryModeldiff():
    init_data = {
        'a': 7500e3,
        'e': 0.1,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 8000,
        'A': 1.0,
    }
    t = np.linspace(0,10*3600.0, num=10000, dtype=np.float)

    
    p2 = PropagatorOrekit()
    print(p2)
    ecefs2 = p2.get_orbit(t, **init_data)

    p1 = PropagatorOrekit(earth_gravity='Newtonian', radiation_pressure=False, solarsystem_perturbers=[], drag_force=False)
    print(p1)
    ecefs1 = p1.get_orbit(t, **init_data)


    dr = np.sqrt(np.sum((ecefs1[:3,:] - ecefs2[:3,:])**2, axis=0))
    dv = np.sqrt(np.sum((ecefs1[3:,:] - ecefs2[3:,:])**2, axis=0))

    r1 = np.sqrt(np.sum(ecefs1[:3,:]**2, axis=0))
    r2 = np.sqrt(np.sum(ecefs1[:3,:]**2, axis=0))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(311)
    ax.plot(t/3600.0, dr*1e-3)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Position difference [km]')
    ax.set_title('Propagation difference diameter simple vs advanced models')
    ax = fig.add_subplot(312)
    ax.plot(t/3600.0, dv*1e-3)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Velocity difference [km/s]')
    ax = fig.add_subplot(313)
    ax.plot(t/3600.0, r1*1e-3, color="green",label='Simple model')
    ax.plot(t/3600.0, r2*1e-3, color="red",label='Advanced model')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Distance from Earth center [km]')
    plt.legend()


    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],"-",alpha=0.5,color="green",label='Simple model')
    ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],"-",alpha=0.5,color="red",label='Advanced model')
    plt.legend()
    plt.show()



def _gen_orbits(num):
    R_E = 6353.0e3
    
    a = R_E*2.0
    orb_init_list = []

    orb_init_list.append(np.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([R_E*1.2, 0, 0.0, 0.0, 0.0, 270], dtype=np.float))
    orb_init_list.append(np.array([R_E*1.2, 1e-9, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 0.0, 0.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 120.0, 0.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([R_E*1.2, 0.1, 0.0, 0.0, 35.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 120.0, 0.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 0.0, 35.0, 0.0], dtype=np.float))
    orb_init_list.append(np.array([R_E*1.2, 0.1, 75.0, 120.0, 35.0, 0.0], dtype=np.float))

    np.random.seed(12398774)

    orb_range = np.array([a, 0.9, 180, 360, 360, 360], dtype=np.float)
    orb_offset = np.array([R_E*1.1, 0, 0.0, 0.0, 0.0, 0.0], dtype=np.float)
    while len(orb_init_list) < num:
        orb = np.random.rand(6)
        orb = orb_offset + orb*orb_range
        if orb[0]*(1.0 - orb[1]) > R_E+200e3:
            orb_init_list.append(orb)

    np.random.seed(None)

    return orb_init_list


def tryTest1():
    init_data = {
            'a': 7500e3,
            'e': 0,
            'inc': 90.0,
            'raan': 10,
            'aop': 10,
            'mu0': 40.0,
            'mjd0': 57125.7729,
            'C_D': 2.3,
            'C_R': 1.0,
            'm': 8000,
            'A': 1.0,
        }

    mjd0 = dpt.jd_to_mjd(2457126.2729)

    orb_init_list = _gen_orbits(100)

    prop = PropagatorOrekit(
            in_frame='EME',
            out_frame='EME',
    )

    t = np.linspace(0, 12*3600, num=100, dtype=np.float)
    
    for kep in orb_init_list:
        state_ref = dpt.kep2cart(kep, m=init_data['m'], M_cent=prop.M_earth, radians=False)

        state_kep = prop.get_orbit(
            t=t, mjd0=mjd0,
            a=kep[0], e=kep[1], inc=kep[2],
            raan=kep[4], aop=kep[3], mu0=dpt.true2mean(kep[5], kep[1], radians=False),
            C_D=init_data['C_D'], m=init_data['m'], A=init_data['A'],
            C_R=init_data['C_R'],
            radians=False,
        )
        state_cart = prop.get_orbit_cart(
            t=t, mjd0=mjd0,
            x=state_ref[0], y=state_ref[1], z=state_ref[2],
            vx=state_ref[3], vy=state_ref[4], vz=state_ref[5],
            C_D=init_data['C_D'], m=init_data['m'], A=init_data['A'],
            C_R=init_data['C_R'],
        )

        state_diff1 = np.abs(state_kep - state_cart)

        try:
            nt.assert_array_less(state_diff1[:3,:], np.full((3,t.size), 1e-5, dtype=state_diff1.dtype))
            nt.assert_array_less(state_diff1[3:,:], np.full((3,t.size), 1e-7, dtype=state_diff1.dtype))
            
        except:

            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot(211)
            ax.plot(t/3600.0, np.sqrt(np.sum(state_diff1[:3,:],axis=0))*1e-3)
            ax.set_xlabel('Time [h]')
            ax.set_ylabel('Position difference [km]')
            ax.set_title('Propagation difference diameter simple vs advanced models')
            ax = fig.add_subplot(212)
            ax.plot(t/3600.0, np.sqrt(np.sum(state_diff1[3:,:],axis=0))*1e-3)
            ax.set_xlabel('Time [h]')

            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot(111, projection='3d')
            plothelp.draw_earth_grid(ax)
            ax.plot(state_kep[0,:], state_kep[1,:], state_kep[2,:],"-",alpha=0.5,color="green",label='KEP input')
            ax.plot(state_cart[0,:], state_cart[1,:], state_cart[2,:],"-",alpha=0.5,color="red",label='CART input')
            plt.legend()
            
            plt.show()


def tryBackwards():

    p = PropagatorOrekit(
        in_frame='ITRF',
        out_frame='ITRF',
        #earth_gravity='Newtonian',
        #radiation_pressure=False,
        #solarsystem_perturbers=[],
        #drag_force=False,
    )

    init_data = {
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 3.0,
        'A': np.pi*(0.5*0.5)**2,
    }

    state0 = [2721793.785377, 1103261.736653, 6427506.515945, 6996.001258,  -171.659563, -2926.43233]
    x, y, z, vx, vy, vz = state0

    t = np.linspace(0,24*3600.0, num=1000, dtype=np.float)

    ecefs1 = p.get_orbit_cart(t, x, y, z, vx, vy, vz, **init_data)

    x, y, z, vx, vy, vz = ecefs1[:,-1].tolist()
    init_data['mjd0'] += t[-1]/(3600.0*24.0)

    print(ecefs1[:,-1])
    ecefs2 = p.get_orbit_cart(-t, x, y, z, vx, vy, vz, **init_data)
    print(ecefs2[:,0])

    print('Position diff: {} m'.format(ecefs1[:3,0] - ecefs2[:3,-1]))
    print('Velocity diff: {} m/s'.format(ecefs1[3:,0] - ecefs2[3:,-1]))

    dr = np.sqrt(np.sum((ecefs1[:3,::-1] - ecefs2[:3,:])**2, axis=0))
    dv = np.sqrt(np.sum((ecefs1[3:,::-1] - ecefs2[3:,:])**2, axis=0))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(211)
    ax.plot(t/3600.0, dr*1e-3)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Position difference [km]')
    ax.set_title('Propagation backwards and forwards error')
    ax = fig.add_subplot(212)
    ax.plot(t/3600.0, dv*1e-3)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Velocity difference [km/s]')


    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],"-",color="green", alpha=0.5,label='Forwards')
    ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],"-",color="red", alpha=0.5,label='Backwards')
    plt.legend()
    plt.show()

def trySentinel1(**kw):

    p = PropagatorOrekit(
        in_frame='ITRF',
        out_frame='ITRF',
        frame_tidal_effects=True,
        #earth_gravity='Newtonian',
        #radiation_pressure=False,
        #solarsystem_perturbers=[],
        #drag_force=False,
    )

    sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')

    x,  y,  z  = sv[0].pos
    vx, vy, vz = sv[0].vel
    mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

    N = len(sv)
    t = 10*np.arange(N)

    kwargs = dict(m=2300., C_R=0., C_D=.0, A=4*2.3)
    kwargs.update(**kw)

    pv = p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0, **kwargs)

    perr = np.linalg.norm(pv[:3].T - sv.pos, axis=1)
    verr = np.linalg.norm(pv[3:].T - sv.vel, axis=1)


    f, ax = plt.subplots(2,1)
    ax[0].plot(t/3600., perr)
    ax[0].set_title('Errors from propagation')
    ax[0].set_ylabel('Position error [m]')
    ax[1].plot(t/3600., verr)
    ax[1].set_ylabel('Velocity error [m/s]')
    ax[1].set_xlabel('Time [h]')




def trySentinel1_array(**kw):

    p = PropagatorOrekit(
        in_frame='ITRF',
        out_frame='ITRF',
        frame_tidal_effects=True,
        #earth_gravity='Newtonian',
        #radiation_pressure=False,
        #solarsystem_perturbers=[],
        #drag_force=False,
    )

    sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')

    x,  y,  z  = sv[0].pos
    vx, vy, vz = sv[0].vel
    mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

    N = len(sv)
    t = 10*np.arange(N)

    num = []
    for _, item in kw.items():
        num.append(len(item))
    num = np.sum(num)/float(len(num))
    if np.abs(num - np.round(num)) < 1e-9:
        num = int(num)
    else:
        raise Exception('All vectors not equal length')

    perr = np.empty((N,num), dtype=np.float64)
    verr = np.empty((N,num), dtype=np.float64)

    kwargs = dict(m=2300., C_R=0., C_D=.0, A=4*2.3)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    pool = comm.size

    my_range = list(range(rank, num, pool))

    for place, ind in enumerate(my_range):
        print('PID{}: {}/{} orbits done'.format(rank, place, len(my_range)))

        for key, item in kw.items():
            kwargs.update({key: item[ind]})

        pv = p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0, **kwargs)

        perr[:,ind] = np.linalg.norm(pv[:3].T - sv.pos, axis=1)
        verr[:,ind] = np.linalg.norm(pv[3:].T - sv.vel, axis=1)


    if pool > 1:
        
        if rank == 0:
            print('---> Thread {}: Receiving all results <barrier>'.format(rank))
            for T in range(1,pool):
                for ind in range(T, num, pool):
                    perr[:,ind] = comm.recv(source=T, tag=ind)
                    verr[:,ind] = comm.recv(source=T, tag=ind*10)
        else:
            print('---> Thread {}: Distributing all filter results to thread 0 <barrier>'.format(rank))
            for ind in my_range:
                comm.send(perr[:,ind], dest=0, tag=ind)
                comm.send(verr[:,ind], dest=0, tag=ind*10)
        print('---> Distributing done </barrier>')

    if rank == 0:
        f, ax = plt.subplots(2,1)
        for ind in range(num):
            ax[0].plot(t/3600., perr[:,ind], '-b', alpha=0.01)
            ax[1].plot(t/3600., verr[:,ind], '-b', alpha=0.01)

        ax[0].set_title('Errors from propagation')
        ax[0].set_ylabel('Position error [m]')
        ax[1].set_ylabel('Velocity error [m/s]')
        ax[1].set_xlabel('Time [h]')
        plt.show()



def trySentinel1_compare(ax, sv, label, **kw):

    prop_args = dict(
        frame_tidal_effects=False,
        radiation_pressure=False,
    )
    for key, item in kw.items():
        if key in prop_args:
            prop_args[key] = item
            del kw[key]

    p = PropagatorOrekit(
        in_frame='ITRF',
        out_frame='ITRF',
        **prop_args
    )

    x,  y,  z  = sv[0].pos
    vx, vy, vz = sv[0].vel
    mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

    N = len(sv)
    t = 10*np.arange(N)

    kwargs = dict(m=2300., C_R=0., C_D=.0, A=4*2.3)
    kwargs.update(**kw)

    pv = p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0, **kwargs)

    perr = np.linalg.norm(pv[:3].T - sv.pos, axis=1)
    verr = np.linalg.norm(pv[3:].T - sv.vel, axis=1)
    
    ax[0].plot(t/3600., perr, label=label)
    ax[1].plot(t/3600., verr, label=label)

    return pv

def compare_effects_orekit():
    f, ax = plt.subplots(2,1)

    sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')

    trySentinel1_compare(ax, sv,
        'No tidal, No PR',
        C_D = 2.3, 
        C_R = 1.0,
        frame_tidal_effects=False,
        radiation_pressure=False,
    )
    trySentinel1_compare(ax, sv,
        'No tidal, PR',
        C_D = 2.3, 
        C_R = 1.0,
        frame_tidal_effects=False,
        radiation_pressure=True,
    )
    trySentinel1_compare(ax, sv,
        'Tidal, PR',
        C_D = 2.3, 
        C_R = 1.0,
        frame_tidal_effects=True,
        radiation_pressure=True,
    )
    ax[0].set_title('Errors from propagation: tidal and radiation effect')
    ax[0].set_ylabel('Position error [m]')
    ax[1].set_ylabel('Velocity error [m/s]')
    ax[1].set_xlabel('Time [h]')
    ax[0].legend()
    plt.show()

def compare_C_D_orekit():
    
    sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')
    N = len(sv)
    t = 10*np.arange(N)

    f, ax = plt.subplots(2,1)
    trySentinel1_compare(ax, sv,
        'C_D = 1.6',
        C_D = 1.6, 
        C_R = 1.0,
        radiation_pressure=True,
    )
    pv = trySentinel1_compare(ax, sv,
        'C_D = 2.3',
        C_D = 2.3, 
        C_R = 1.0,
        radiation_pressure=True,
    )
    trySentinel1_compare(ax, sv,
        'C_D = 3.0',
        C_D = 3.0, 
        C_R = 1.0,
        radiation_pressure=True,
    )
    ax[0].set_title('Errors from propagation: drag coefficient comparison')
    ax[0].set_ylabel('Position error [m]')
    ax[1].set_ylabel('Velocity error [m/s]')
    ax[1].set_xlabel('Time [h]')

    '''
    ax2 = ax[0].twinx()

    ax2.plot(t/3600., np.linalg.norm(pv[:3].T, axis=1), 'b.', alpha=0.25)
    ax2.set_ylabel('Orbital radius', color='b')
    ax2.tick_params('y', colors='b')
    '''
    f.tight_layout()

    ax[0].legend()
    plt.show()


def compare_grav_orekit():
    f, ax = plt.subplots(2,1)

    sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')

    trySentinel1_compare(ax, sv,
        'Newtonian grav',
        C_D = 2.3, 
        C_R = 1.0,
        radiation_pressure=True,
        earth_gravity='Newtonian',
    )
    trySentinel1_compare(ax, sv,
        'HolmesFeatherstone grav',
        C_D = 2.3, 
        C_R = 1.0,
        radiation_pressure=True,
        earth_gravity='HolmesFeatherstone',
    )
    ax[0].set_title('Errors from propagation: gravity model')
    ax[0].set_ylabel('Position error [m]')
    ax[1].set_ylabel('Velocity error [m/s]')
    ax[1].set_xlabel('Time [h]')
    ax[0].legend()
    plt.show()


def compare_perturbers_orekit():
    f, ax = plt.subplots(2,1)

    sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')

    trySentinel1_compare(ax, sv,
        'no solarsystem perturbers',
        C_D = 2.3, 
        C_R = 1.0,
        radiation_pressure=True,
        solarsystem_perturbers=[],
    )
    trySentinel1_compare(ax, sv,
        'Moon + sun perturbers',
        C_D = 2.3, 
        C_R = 1.0,
        radiation_pressure=True,
        solarsystem_perturbers=['Moon', 'Sun'],
    )
    ax[0].set_title('Errors from propagation: Third body perturbations comparison')
    ax[0].set_ylabel('Position error [m]')
    ax[1].set_ylabel('Velocity error [m/s]')
    ax[1].set_xlabel('Time [h]')
    ax[0].legend()
    plt.show()

def tryPolarOrbit():

    _prop = propagator_orekit.PropagatorOrekit
    _opts = {
        'in_frame': 'TEME',
        'out_frame': 'ITRF',
        'solar_activity_strength': 'WEAK',
    }

    prop = _prop(**_opts)

    init_data = {
        'a': (R_e + 400.0)*1e3,
        'e': 0.01,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 54832.0,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 3.0,
        'A': np.pi*(0.1)**2,
    }
    t = np.linspace(0,48*3600.0, num=5000, dtype=np.float)

    ecefs = prop.get_orbit(t, **init_data)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs[0,:], ecefs[1,:], ecefs[2,:],".",color="green")
    return fig, ax



def tryRangeRate(prop):

    init_data = {
        'a': (R_e + 400.0)*1e3,
        'e': 0.01,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 54832.0,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 3.0,
        'A': np.pi*(0.1)**2,
    }
    t = 5.6*3600.0
    dt = 1.0

    ecefs = prop.get_orbit(t, **init_data)
    ecefs_p = prop.get_orbit(t+dt, **init_data)
    ecefs_m = prop.get_orbit(t-dt, **init_data)

    station = np.array([0, 0, 7000e3], dtype=np.float64)

    print('-'*20)
    print(prop)

    print('Prop vel:')
    print(ecefs[3:])
    prop_vel = ecefs[3:]

    num_vel = 0.5*(ecefs_p[:3] - ecefs[:3] + ecefs[:3] - ecefs_m[:3])/dt

    print('num vel:')
    print(num_vel)

    print('Velocity difference: {} m/s'.format(np.linalg.norm(prop_vel - num_vel)))

    return ecefs[3:], num_vel

def compareRangeRates():
    import propagator_sgp4

    _prop = propagator_orekit.PropagatorOrekit
    _opts = {
        'in_frame': 'TEME',
        'out_frame': 'ITRF',
        'solar_activity_strength': 'WEAK',
    }

    prop = _prop(**_opts)

    tryRangeRate(prop)

    _prop = propagator_sgp4.PropagatorSGP4
    _opts = {
        'out_frame': 'ITRF',
        'polar_motion': True,
    }

    prop = _prop(**_opts)

    tryRangeRate(prop)





def tryLogging(t):
    def print_(s):
        print(s)

    p = PropagatorOrekit(logger_func=print_)

    init_data = {
        'a': (R_e + 400.0)*1e3,
        'e': 0.01,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 54832.0,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 3.0,
        'A': np.pi*(0.1)**2,
    }

    ecefs = p.get_orbit(t, **init_data)

    p.print_logger()
    return p.get_logger()


if __name__=='__main__':
    #trySteps()
    #tryActivity()

    #compare_C_D_orekit()
    #compare_effects_orekit()
    #compare_grav_orekit()
    #compare_perturbers_orekit()

    #trySentinel1(C_D=2.3, C_R = 1.0)
    #trySentinel1_array(C_D=np.linspace(1.5,3.5,num=100,dtype=np.float64))
    #tryBackwards()

    #try1()
    #try2()
    #tryMPI()
    #trySpeed()
    #tryFrame()
    #tryModeldiff()
    #tryTest1()

    #tryPolarOrbit()

    '''
    print('='*15 + ' 10 s step for 24 h ' + '='*15)
    tryLogging(np.arange(0, 24*3600.0, 10.0))

    print('='*15 + ' 1 h step for 24 h ' + '='*15)
    tryLogging(np.arange(0, 24*3600.0, 3600.0))

    print('='*15 + ' 10 s step for 1 h ' + '='*15)
    tryLogging(np.arange(0, 3600.0, 10.0))
    '''

    tryBackwards()
    #compareRangeRates()



    plt.show()