#!/usr/bin/env python
#
#
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from mpi4py import MPI
import sys
import os
import time

comm = MPI.COMM_WORLD

# replace this with the path to your sorts
sys.path.insert(0, "/home/danielk/IRF/IRF_GITLAB/SORTSpp")
#sys.path.insert(0, "/home/danielk/PYTHON/SORTSpp")

# SORTS imports CORE
import population_library as plib
from simulation import Simulation
from population import Population

#SORTS Libraries
import radar_library as rlib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib

import propagator_sgp4
import propagator_orekit
import dpt_tools as dpt
import plothelp

from catalogue import Catalogue

os.environ['TZ'] = 'GMT'
time.tzset()

####### RUN CONFIG #######
campagin = 0
##########################


if campagin == 0:
    the_date = '_2019_04_02_'
    SIM_TIME = 24.0
    dt = np.datetime64('2019-04-02T12:01')
    mjd = dpt.npdt2mjd(dt)
    scan = rslib.beampark_model(
        lat = radar._tx[0].lat,
        lon = radar._tx[0].lon,
        alt = radar._tx[0].alt, 
        az = 90.0,
        el = 70.0,
    )
elif campagin == 1:
    the_date = '_2019_04_05_'
    SIM_TIME = 5.0
    dt = np.datetime64('2019-04-05T08:01')
    mjd = dpt.npdt2mjd(dt)
    scan = rslib.beampark_model(
        lat = radar._tx[0].lat,
        lon = radar._tx[0].lon,
        alt = radar._tx[0].alt, 
        az = 90.0,
        el = 45.0,
    )
branch_name = 'deb_simulation' + the_date

_prop = propagator_orekit.PropagatorOrekit
_opts = {
    'in_frame': 'TEME',
    'out_frame': 'ITRF',
    'solar_activity_strength': 'WEAK',
}

event_date = np.datetime64('2019-03-27T05:40')
event_mjd = dpt.npdt2mjd(event_date)

prop = _prop(**_opts)

#sim_root = '/ZFS_DATA/SORTSpp/FP_sims/T_UHF_MicrosatR_v2'
sim_root = '/home/danielk/IRF/E3D_PA/FP_sims/T_UHF_MicrosatR_v2'

#initialize the radar setup 
radar = rlib.eiscat_uhf()

radar.set_FOV(max_on_axis=25.0, horizon_elevation=50.0)
radar.set_SNR_limits(min_total_SNRdb=14.0, min_pair_SNRdb=14.0)

radar.set_scan(scan)

fname = sim_root + '/population_{}.h5'.format(branch_name)
catname = sim_root + '/' + branch_name + '/catalogue_data.h5'

pop = Population.load(fname,
    propagator = _prop,
    propagator_options = _opts,
)

cat = Catalogue.from_file(pop, catname)

time_step = 80.0
#time_step = 600.0

t_end = (mjd - event_mjd + SIM_TIME/24.0)*3600.0*24.0
t_meas = (mjd - event_mjd)*3600.0*24.0
#_deb_tt = np.arange(0, 3600.0*12.0, time_step, dtype=np.float64)
_deb_tt = np.arange(0, 3600.0*6.0, time_step, dtype=np.float64)
#_mes_tt = np.arange(t_meas, t_end, time_step, dtype=np.float64)
#tt = np.append(
#    _deb_tt,
#    _mes_tt,
#)
tt = _deb_tt


detected_index = np.array([track['index'] for track in cat.tracks if track['tracklet']], dtype=np.int64)
detected_index = np.unique(detected_index).tolist()

run_index = detected_index

#print(detected_index)
#exit()

#_del = [ind for ind in range(len(pop)) if ind not in detected_index]
#pop.delete(_del)
#pop.delete(slice(3,None))
#detected_index = list(range(len(pop)))
#for ind in detected_index:
#    cat.tracklets[ind]['index'] = ind

state_TEME, sat_id = plib._get_MicrosatR_state(event_mjd)

states = np.empty((6,len(tt), len(pop)), dtype=np.float64)

_tstart = time.time()

dets = {'t': [], 'r': [], 'v': []}

prop_ok = np.full((len(pop),), True, np.bool)

my_inds = list(range(comm.rank, len(pop), comm.size))

for ind in my_inds:
    if ind not in run_index:
        continue
    pert_state = state_TEME[:,0].copy()

    dv = pop['dV'][ind]
    ddir = np.empty((3,), dtype=np.float64)
    ddir[0] = pop.objs[ind]['dx']
    ddir[1] = pop.objs[ind]['dy']
    ddir[2] = pop.objs[ind]['dz']

    pert_state[3:] += ddir*dv

    prop_data = dict(
        x = pert_state[0],
        y = pert_state[1],
        z = pert_state[2],
        vx = pert_state[3],
        vy = pert_state[4],
        vz = pert_state[5],
        mjd0 = event_mjd,
        t = tt,
    )

    prop_data['A'] = pop.objs[ind]['A'] 
    prop_data['d'] = pop.objs[ind]['d'] 
    prop_data['m'] = pop.objs[ind]['m'] 
    prop_data['C_D'] = pop.objs[ind]['C_D']
    prop_data['C_R'] = pop.objs[ind]['C_R']

    try:
        states[:,:,ind] = prop.get_orbit_cart(**prop_data)
    except:
        prop_ok[ind] = False
        continue

    if ind in detected_index:
        print('Object detected, generating measurement data')

        station = radar._tx[0].ecef

        for tracklet in cat.tracklets:
            if tracklet['index'] == ind:

                s_obj = pop.get_object(tracklet['index'])
                tmp_states = s_obj.get_state(tracklet['t'])
                num = len(tracklet['t'])
                rel_pos = tmp_states[:3,:]
                rel_vel = tmp_states[3:,:]
                range_v = np.empty((num,), dtype=np.float64)
                velocity_v = np.empty((num,), dtype=np.float64)
                for _ind in range(num):
                    rel_pos[:,_ind] -= station

                    range_v[_ind] = np.linalg.norm(rel_pos[:,_ind])
                    velocity_v[_ind] = -np.dot(rel_pos[:,_ind],rel_vel[:,_ind])/range_v[_ind]

                dets['t'] += tracklet['t'].tolist()
                dets['r'] += (range_v*1e-3).tolist()
                dets['v'] += (velocity_v*1e-3).tolist()

    _telaps = time.time() - _tstart
    _tleft = _telaps/float(ind+1)*float(len(pop) - ind + 1)
    print('PID{}: Object {} / {} propagated {} steps to {:.2f} h. Time left {:.2f} h'.format(comm.rank, ind+1, len(pop), len(tt), t_end/3600.0, _tleft/3600.0))

if comm.rank == 0:
    for thr_id in range(1, comm.size):
        dets['t'] += comm.recv(source=thr_id, tag=len(pop)*10+1)
        dets['r'] += comm.recv(source=thr_id, tag=len(pop)*10+2)
        dets['v'] += comm.recv(source=thr_id, tag=len(pop)*10+3)
        for ind in range(thr_id, len(pop), comm.size):
            states[:,:,ind] = comm.recv(source=thr_id, tag=ind*10+1)
            prop_ok[ind] = comm.recv(source=thr_id, tag=ind*10+2)
else:
    comm.send(dets['t'], dest=0, tag=len(pop)*10+1)
    comm.send(dets['r'], dest=0, tag=len(pop)*10+2)
    comm.send(dets['v'], dest=0, tag=len(pop)*10+3)
    for ind in my_inds:
        comm.send(states[:,:,ind], dest=0, tag=ind*10+1)
        comm.send(prop_ok[ind], dest=0, tag=ind*10+2)

if comm.rank != 0:
    exit()

pop.objs = pop.objs[prop_ok]
states = states[:,:, prop_ok]

dets['t'] = np.array(dets['t'])
dets['r'] = np.array(dets['r'])
dets['v'] = np.array(dets['v'])

plt.style.use('dark_background')

fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(111, projection='3d')
plothelp.draw_earth_grid(ax, alpha = 0.2, color='white')

ax.grid(False)
plt.axis('off')



#traj
ax_traj_list = []
ax_point_list = []
for ind in range(len(pop)):
    ax_traj, = ax.plot([],[],[], '-', alpha=0.5, color="white")
    ax_traj_list.append(ax_traj)

    ax_point, = ax.plot([],[],[], '.', alpha=1, color="blue")
    ax_point_list.append(ax_point)

#init
ecef_point, k0 = radar._tx[0].scan.antenna_pointing(0.0)

beam_range = 3000e3

track_len = 25

#radar beam
ax_txb, = ax.plot(
    [ecef_point[0],ecef_point[0]],
    [ecef_point[1],ecef_point[1]],
    [ecef_point[2],ecef_point[2]],
    alpha=1,color="green",
)

ax.plot(
    [ecef_point[0]],
    [ecef_point[1]],
    [ecef_point[2]],
    'or',
    alpha=1,
)

dets_ax = fig.add_axes([.65, .05, .33, .25], facecolor='k')
dets_ax_points, = dets_ax.plot([], [], '.r', alpha = 1)
dets_ax_time, = dets_ax.plot([], [], '-b', alpha = 1)

dets_ax.set(
    ylabel='Range [km]',
    xlabel='Time past $t_0$ [h]',
    title='Simulated Microsat-R detections',
)
_ylim = 2000.0

dets_ax.set_ylim(0, _ylim)
dets_ax.set_xlim(0, SIM_TIME)

def run(tt_ind):
    # update the data

    titl.set_text('Simulation: ASAT event t+ %.4f h' % (tt[tt_ind]/3600.0))

    if (tt_ind - track_len) < 0:
        tt_prev = 0
    elif tt_ind > len(_deb_tt) and (tt_ind - track_len) <= len(_deb_tt):
        tt_prev = len(_deb_tt)
    else:
        tt_prev = tt_ind - track_len

    print('Updating plot frame {} of {}'.format(tt_ind, len(tt)))
    for ind, ax_traj in enumerate(ax_traj_list):

        if ind not in run_index:
            continue

        ax_traj.set_data(
            states[0, tt_prev:(tt_ind+1), ind],
            states[1, tt_prev:(tt_ind+1), ind],
        )
        ax_traj.set_3d_properties(
            states[2, tt_prev:(tt_ind+1), ind],
        )
        ax_traj.figure.canvas.draw()

        ax_point_list[ind].set_data(
            states[0, tt_ind, ind],
            states[1, tt_ind, ind],
        )
        ax_point_list[ind].set_3d_properties(
            states[2, tt_ind, ind],
        )
        ax_point_list[ind].figure.canvas.draw()

    if (t_end - tt[tt_ind])/3600.0 < SIM_TIME:
        ax_txb.set_data(
            [ecef_point[0],ecef_point[0] + k0[0]*beam_range],
            [ecef_point[1],ecef_point[1] + k0[1]*beam_range],
        )
        ax_txb.set_3d_properties(
            [ecef_point[2],ecef_point[2] + k0[2]*beam_range],
        )
        ax_txb.figure.canvas.draw()
    else:
        ax_txb.set_data(
            [ecef_point[0],ecef_point[0] + k0[0]*0],
            [ecef_point[1],ecef_point[1] + k0[1]*0],
        )
        ax_txb.set_3d_properties(
            [ecef_point[2],ecef_point[2] + k0[2]*0],
        )
        ax_txb.figure.canvas.draw()

    t_rel = (tt[tt_ind] - t_meas)
    inds = dets['t'] < t_rel
    dets_ax_points.set_data(
        dets['t'][inds]/3600.0,
        dets['r'][inds],
    )
    if t_rel > 0:
        dets_ax_time.set_data(
            [t_rel/3600.0, t_rel/3600.0],
            [0.0, _ylim],
        )

    return ax_traj_list, ax_point_list, ax_txb, dets_ax_points, dets_ax_time


delta = 7500e3
ax.set_xlim([-delta,delta])
ax.set_ylim([-delta,delta]) 
ax.set_zlim([-delta,delta]) 


titl = fig.text(0.5,0.94,'',size=22,horizontalalignment='center')


ani = animation.FuncAnimation(fig, run, range(len(tt)),
    blit=False,
    interval=50.0,
    repeat=True,
)

print('Anim done, writing movie')

fps = 20

Writer = animation.writers['ffmpeg']
writer = Writer(metadata=dict(artist='Daniel Kastinen'),fps=fps)
ani.save(sim_root + '/MicrosatR_movie_v2.mp4', writer=writer)

print('showing plot')
plt.show()
