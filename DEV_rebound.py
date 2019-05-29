#!/usr/bin/env python

from mpi4py import MPI
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import plothelp

import dpt_tools as dpt
import orbit_verification as over
import radar_library as rlib
import population_library as plib
from population import Population

from propagator_rebound import PropagatorRebound
R_e = 6371.0

radar = rlib.eiscat_3d(beam='interp', stage=1)

pop = plib.NESCv9_mini_moons(albedo = 0.14, propagate_SNR = 1.0, radar = radar, truncate = slice(1,None))
#pop = plib.NESCv9_mini_moons(albedo = 0.14)

#plt.hist(pop['d'])
#plt.show()

exit()

def try2():

    p = PropagatorRebound(
        in_frame = 'EME',
        out_frame = 'ITRF',
    )
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
        'A': np.pi*(2.0*0.5)**2,
    }
    t = np.linspace(0,1.0*24*3600.0, num=1000, dtype=np.float)

    ecefs1 = p.get_orbit(t, **init_data)

    max_range = init_data['a']*1.1

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],"-b")
    
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    plt.show()


def gen_pop():
    pop = plib.NESCv9_mini_moons(num = 1, d_range = [0.1, 10.0], synchronize=True)
    pop.save('./data/NESCv9_mult1.h5')

def plot_mini_moons_propagation(d = 10.0):

    #pop = Population.load('./data/NESCv9_mult1.h5')
    pop = plib.NESCv9_mini_moons(num = 1, d_range = [0.1, 10.0], synchronize=False)

    pop.delete(slice(40,None))

    gen = pop.object_generator()

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot([0], [0], [0],"xr", alpha = 1)

    t_v = np.linspace(0, d*24.0*3600.0, num=1000, dtype=np.float)

    for sobj in gen:
        states = sobj.get_orbit(t_v)
        ax.plot(states[0,:], states[1,:], states[2,:],"-b", alpha = 0.05)

    max_range = 384400e3*1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    plt.show()



def plot_mini_moons_range(d = 300.0):

    pop = plib.NESCv9_mini_moons(num = 1, d_range = [0.1, 10.0], synchronize=False)

    pop.delete(slice(40,None))

    gen = pop.object_generator()

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)

    t_v = np.linspace(0, d*24.0*3600.0, num=1000, dtype=np.float)

    for ind, sobj in enumerate(gen):
        states = sobj.get_orbit(t_v)
        print('obj {} of {}'.format(ind, len(pop)))
        ax.plot(t_v/(24.0*3600.0), np.linalg.norm(states, axis=0)/384400e3,"-b", alpha = 0.2)
    
    ax.set(
        title='Distance from Earth as a function of time from epoch',
        ylabel='Distance [Lunar distances]',
        xlabel='Days from epoch [d]',
    )
    plt.show()



def plot_mini_moons_EME(d = 200.0):

    #pop = Population.load('./data/NESCv9_mult1.h5')
    pop = plib.NESCv9_mini_moons(num = 1, d_range = [0.1, 10.0], synchronize=False)

    pop.delete(slice(100,None))

    pop.propagator_options['out_frame'] = 'ITRF'
    pop.propagator_options['time_step'] = 3600.0
    gen = pop.object_generator()

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    
    t_v = np.linspace(0,d*24.0*3600.0, num=1000, dtype=np.float)
    
    for ind, sobj in enumerate(gen):
        #print(sobj)
        print('obj {} of {}'.format(ind, len(pop)))
        states = sobj.get_orbit(t_v)
        ax.plot(states[0,:], states[1,:], states[2,:],"-b", alpha = 0.1)

    plt.show()



def plot_mini_moons_solarsystem(d = 200.0):

    #pop = Population.load('./data/NESCv9_mult1.h5')
    pop = plib.NESCv9_mini_moons(num = 1, d_range = [0.1, 10.0], synchronize=False)

    pop.delete(slice(100,None))

    pop.propagator_options['out_frame'] = 'HeliocentricEME'
    pop.propagator_options['time_step'] = 3600.0
    gen = pop.object_generator()

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    
    t_v = np.linspace(0,d*24.0*3600.0, num=1000, dtype=np.float)
    
    for ind, sobj in enumerate(gen):
        #print(sobj)
        print('obj {} of {}'.format(ind, len(pop)))
        states = sobj.get_orbit(t_v)
        ax.plot(states[0,:], states[1,:], states[2,:],"-b", alpha = 0.01)

    plt.show()

#plot_mini_moons_solarsystem()
#plot_mini_moons_propagation(d=10.0)
#plot_mini_moons_EME()

plot_mini_moons_range()
exit()


#SORTS Libraries
import radar_library as rlib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib

import simulate_tracking

####### RUN CONFIG #######
part = 1
SIM_TIME = 3600.0*24.0*30.0
##########################

sim_root = '/ZFS_DATA/SORTSpp/FP_sims/E3D_scanning'

#initialize the radar setup 
radar = rlib.eiscat_3d(beam='interp', stage=1)

radar.set_FOV(max_on_axis=90.0, horizon_elevation=30.0)
radar.set_SNR_limits(min_total_SNRdb=1.0, min_pair_SNRdb=1.0)
radar.set_TX_bandwith(bw = 1.0e6)

pop = plib.NESCv9_mini_moons(num = 1, d_range = [0.1, 2.0], synchronize=False)
pop.delete(slice(100,None))

gen = pop.object_generator()
for sobj in gen:
    pass_struct = simulate_tracking.get_passes(sobj, radar, 0.0, SIM_TIME, t_samp=60.0)
    print(pass_struct)
#try2()