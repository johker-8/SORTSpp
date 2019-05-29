#!/usr/bin/env python

'''
Ide: behover gora en continuelerig model for comet orbit: propagera orbit och sen skapa chebychev polynomials mellan punkter for att gora en smooth function.
'''
import os

import numpy as np
import scipy
import h5py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import dpt_tools as dpt

import rebound
import spiceypy as spice

spice.furnsh("./data/spice/MetaK.txt")

AU = 149597871e3


plot = True

if plot:
    plt.ion()

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-2*AU, 2*AU)
    ax.set_ylim(-2*AU, 2*AU)
    ax.set_zlim(-2*AU, 2*AU)

lin_n = 0



class ReboundMOID(object):

    def __init__(self,
                integrator='IAS15',
                time_step=3600.0,
                mjd0 = 53005,
                min_MOID = 0.01*AU,
            ):
        self.min_MOID = min_MOID
        self.planets = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
        self.planets_mass = [0.330104e24, 4.86732e24, 5.97219e24, 0.641693e24, 1898.13e24, 568.319e24, 86.8103e24, 102.410e24]
        self.m_sun = 1.98855e30
        self._earth_ind = 3
        self.N_massive = len(self.planets) + 1;
        self.integrator = integrator
        self.time_step = time_step
        self.mjd0 = mjd0
        self.states = np.empty((6,0), dtype=np.float64)
        self.t0 = 0.0


    def _setup_sim(self):
        self.sim = rebound.Simulation()
        self.sim.units = ('m', 's', 'kg')
        self.sim.integrator = self.integrator
        self.et = dpt.mjd_to_j2000(self.mjd0)*3600.0*24.0
        
        self.sim.add(m=self.m_sun)
        for i in range(0,len(self.planets)):
            #Units are always km and km/sec.
            state, lightTime = spice.spkezr(
                self.planets[i] + ' BARYCENTER',
                self.et,
                'J2000',
                'NONE',
                'SUN',
            )
            self.sim.add(m=self.planets_mass[i],
                x=state[0]*1e3,  y=state[1]*1e3,  z=state[2]*1e3,
                vx=state[3]*1e3, vy=state[4]*1e3, vz=state[5]*1e3,
            )
        self.sim.N_active = self.N_massive
        self.sim.dt = self.time_step
        
        for ind in range(self.num):
            
            x, y, z, vx, vy, vz = self.states[:,ind]
    
            self.sim.add(
                x = x,
                y = y,
                z = z,
                vx = vx,
                vy = vy,
                vz = vz,
                m = 0.0,
            )

    @property
    def num(self):
        return self.states.shape[1]
    


    def _get_state(self, ind):
        particle = self.sim.particles[self.N_massive + ind]
        state = np.empty((6,), dtype=np.float64)
        state[0] = particle.x
        state[1] = particle.y
        state[2] = particle.z
        state[3] = particle.vx
        state[4] = particle.vy
        state[5] = particle.vz
        return state

    def _get_earth_state(self):
        earth_state = np.empty((6,), dtype=np.float64)
        earth = self.sim.particles[self._earth_ind]
        earth_state[0] = earth.x
        earth_state[1] = earth.y
        earth_state[2] = earth.z
        earth_state[3] = earth.vx
        earth_state[4] = earth.vy
        earth_state[5] = earth.vz
        return earth_state

    def propagate(self, t, **kwargs):

        self._setup_sim()

        results = []
        for ind in range(self.num):
            results.append({
                'MOID': None,
                't': None,
                'ind': None,
                'state': None,
            })

        self.sim.move_to_com()

        states = np.empty((len(t), 6, self.num))
        states_E = np.empty((len(t), 6))

        for ti in range(len(t)):

            self.sim.integrate(t[ti])
            
            state_E = self._get_earth_state()
            states_E[ti, :] = state_E

            for ind in range(self.num):
                
                state = self._get_state(ind)    
                states[ti, :, ind] = state


                MOID = np.linalg.norm(state - state_E)
            
                tm = results[ind]['t']
                moid0 = results[ind]['MOID']
                if tm is None or moid0 is None:
                    results[ind]['t'] = t[ti]
                    results[ind]['ind'] = ti
                    results[ind]['MOID'] = MOID
                
                if MOID < moid0 and moid0 > self.min_MOID:
                    results[ind]['t'] = t[ti]
                    results[ind]['ind'] = ti
                    results[ind]['MOID'] = MOID
        
        if kwargs.get('plot', False) and plot:

            for p in range(self.num):
                dst[p].set_data(
                    [states_E[results[ind]['ind'], 0], states[results[ind]['ind'], 0, p]],
                    [states_E[results[ind]['ind'], 1], states[results[ind]['ind'], 1, p]],
                )
                dst[p].set_3d_properties(
                    [states_E[results[ind]['ind'], 2], states[results[ind]['ind'], 2, p]],
                )

            global lin_n
            lin[lin_n].set_data(
                states[:, 0, 0],
                states[:, 1, 0],
            )
            lin[lin_n].set_3d_properties(
                states[:, 2, 0],
            )

            if lin_n == 0:
                lin_E[0].set_data(
                    states_E[:, 0],
                    states_E[:, 1],
                )
                lin_E[0].set_3d_properties(
                    states_E[:, 2],
                )

            fig.canvas.draw()
            fig.canvas.flush_events()
            
            lin_n += 1


        return results


fname = '/home/danielk/IRF/E3D_PA/moid_test/data.h5'

if plot:
    lin = []
    dst = []
    lin_E = ax.plot([], [], [], '-b', alpha=1)

def run_moid():

    t_end = 2.0 #in years
    dt = 3600.0*24.0
    _year = 3600*24*365.25
    steps = 1000

    prop = ReboundMOID(time_step=dt)

    prop.states = np.array([
        [1.3*AU, 0.0, 0.5*AU, 0.0, 5.0e3, 23e3],
        [1.3*AU, 0.0, 0.2*AU, 0.0, 18.0e3, 5e3],
    ], dtype=np.float64).T
    prop.t0 = 0.0

    if plot:
        global lin
        global dst
        for ind in range(steps):
            lin += ax.plot([], [], [], '-k', alpha=0.01)
        for ind in range(prop.num):
            dst += ax.plot([], [], [], '-r', alpha=0.7)

    step = np.array([[10e3, 10e3, 10e3, 1e3, 1e3, 1e3]], np.float64)
    step = np.repeat(step.T, prop.num, axis = 1)

    t = np.arange(0,t_end*_year, dt, dtype=np.float64)

    chain = np.zeros((steps, 6, prop.num), dtype=np.float64)
    moids = np.zeros((steps, prop.num), dtype=np.float64)
    accept = np.zeros((prop.num, 6), dtype=np.float64)
    tries = np.zeros((prop.num, 6), dtype=np.float64)

    logpost = np.zeros((prop.num, ), dtype=np.float64)
    logpost_try = np.zeros((prop.num, ), dtype=np.float64)

    results = prop.propagate(t)
    for p in range(prop.num):
        logpost[p] = -results[p]['MOID']
            
    for ind in range(steps):

        x_current = np.copy(prop.states)
        
        for p in range(prop.num):
            pi = int(np.floor(np.random.rand(1)*6))

            prop.states[pi, p] += np.random.randn(1)*step[pi, p]
        
        results = prop.propagate(t, plot=True)

        for p in range(prop.num):
            logpost_try[p] = -results[p]['MOID']
            
            alpha = np.log(np.random.rand(1))
            
            if logpost_try[p] > logpost[p]:
                _accept = True
            elif (logpost_try[p] - alpha) > logpost[p]:
                _accept = True
            else:
                _accept = False
            
            tries[p, pi] += 1.0
            
            if _accept:
                logpost[p] = logpost_try[p]
                accept[p, pi] += 1.0
            else:
                prop.states[:, p] = x_current[:, p]
            
            
            print('log post {:<5.2e} AU (particle {} iter {}/{})'.format(logpost_try[p]/AU, p, ind, steps))

            if ind % 100 == 0 and ind > 0:
                for dim in range(6):
                    ratio = accept[p, dim]/tries[p, dim]

                    print('ratio {:<5.2f}'.format(ratio))                    
                    if ratio > 0.5:
                        step[dim, p] *= 2.0
                    elif ratio < 0.3:
                        step[dim, p] /= 2.0
                    
                    accept[p, dim] = 0.0
                    tries[p, dim] = 0.0
            
            chain[ind, :, :] = prop.states.copy()
            moids[ind, p] = results[p]['MOID']


    with h5py.File(fname, 'w') as hf:
        hf['chain'] = chain
        hf['moids'] = moids



if __name__ == '__main__':

    if not os.path.exists(fname):
        run_moid()
    

    with h5py.File(fname, 'r') as hf:
        chain = hf['chain'].value
        moids = hf['moids'].value


    #anim = animation.ArtistAnimation(fig, images)
    #anim.save('/'.join(fname.split('/')[:-1] + ['z.gif']), writer='imagemagick', fps=20)


    fig, ax = plt.subplots()
    for ind in range(moids.shape[1]):
        ax.plot(moids[:,ind]/AU)

    fig, axes = plt.subplots(2, 3, sharex=True)

    dim = 0
    for xi in range(2):
        for yi in range(3):
            for ind in range(moids.shape[1]):
                axes[xi, yi].plot(chain[:, dim, ind])
            dim += 1


    fig, axes = plt.subplots(3, 3)

    dim = 0
    for xi in range(3):
        for yi in range(3):
            for ind in range(moids.shape[1]):
                axes[xi, yi].plot(chain[:, xi, ind], chain[:, yi+3, ind], '.')

    plt.show()
