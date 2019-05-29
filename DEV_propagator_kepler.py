import time

import numpy as np
import matplotlib.pyplot as plt

import plothelp
from propagator_kepler import PropagatorKepler
from propagator_sgp4 import MU_earth

prop = PropagatorKepler()



def try1():
    init_data = {
        'a': 7000e3,
        'e': 0.05,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 90.0,
        'mjd0': 57125.7729,
        'radians': False,
        'm': 8000,
    }


    period = np.pi*2.0*np.sqrt(init_data['a']**3/MU_earth)

    t = np.linspace(0, 2*period, endpoint=True, num=1000, dtype=np.float)

    print('period: {} hours'.format(period/3600.0))

    t0 = time.time()
    ecefs = prop.get_orbit(t, **init_data)
    print('get orbit time: {} sec'.format(time.time() - t0))


    t0 = time.time()

    state_end = prop.output_to_input_frame(ecefs[:,-1], t[-1], init_data['mjd0'])
    x, y, z, vx, vy, vz = state_end.tolist()

    param_data = {
        'mjd0': init_data['mjd0'] + t[-1]/(3600.0*24.0),
        'radians': False,
        'm': 8000,
    }

    ecefs2 = prop.get_orbit_cart(t, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, **param_data)
    print('get orbit time (cart): {} sec'.format(time.time() - t0))


    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ecefs[0,:], ecefs[1,:], ecefs[2,:],".",color="green")
    ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],".",color="red")
    plt.show()

def tryBackwards():

    prop = PropagatorKepler(in_frame='ITRF', out_frame='ITRF')
    init_data = {
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 3.0,
        'A': np.pi*(0.5*0.5)**2,
    }

    state0 = [2721793.785377, 1103261.736653, 6427506.515945, 6996.001258,  -171.659563, -2926.43233]
    x, y, z, vx, vy, vz = state0

    t = np.linspace(0,24*3600.0, num=5000, dtype=np.float)

    ecefs1 = prop.get_orbit_cart(t, x, y, z, vx, vy, vz, **init_data)

    x, y, z, vx, vy, vz = ecefs1[:,-1].tolist()
    init_data['mjd0'] += t[-1]/(3600.0*24.0)

    ecefs2 = prop.get_orbit_cart(-t, x, y, z, vx, vy, vz, **init_data)

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
    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],"-",color="green", alpha=0.5)
    ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],"-",color="red", alpha=0.5)
    plt.show()



tryBackwards()
try1()