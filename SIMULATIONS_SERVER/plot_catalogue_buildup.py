#!/usr/bin/env python
#
#
import os
import copy
import numpy as np
from mpi4py import MPI
import sys
import h5py

# replace this with the path to your sorts
sys.path.insert(0, "/home/danielk/PYTHON/SORTSpp")

# SORTS imports CORE
import population_library as plib
from simulation import Simulation
from catalogue import Catalogue

#SORTS Libraries
import radar_library as rlib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib

import dpt_tools as dpt

_plot_out = '/ZFS_DATA/SORTSpp/FP_sims/plots'

radar_e3d = rlib.eiscat_3d(beam='interp', stage=1)

radar_e3d.set_FOV(max_on_axis=90.0, horizon_elevation=30.0)
radar_e3d.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=1.0)
radar_e3d.set_TX_bandwith(bw = 1.0e6)

pop_e3d = plib.filtered_master_catalog_factor(
    radar = radar_e3d,
    treshhold = 0.01,
    min_inc = 50,
    prop_time = 48.0,
)


sims_plot = [
#    ('/ZFS_DATA/SORTSpp/FP_sims/E3D_scanning', 'scheduler_2d', pop_e3d),
]

tsr_branches = [
    'MASTER2009_' + 'SGP4' + '_' + '500kW' + '_' + '1.2GHz',
    #'MASTER2009_' + 'SGP4' + '_' + '500kW' + '_' + '2.4GHz',
    #'MASTER2009_' + 'SGP4' + '_' + '100kW' + '_' + '1.2GHz',
]

#initialize the radar setup 
radar = rlib.tromso_space_radar()
tsr_sim_root = '/ZFS_DATA/SORTSpp/FP_sims/TSR_beampark'

for tsr_branch_name in tsr_branches:

    #load the input population
    pop_tsr = plib.filtered_master_catalog_factor(
        radar = radar,  
        treshhold = 0.01,
        min_inc = 50,
        prop_time = 48.0,
        detectability_file = tsr_sim_root + '/' + tsr_branch_name + '.h5',
    )

    sims_plot.append(
        (tsr_sim_root, tsr_branch_name, pop_tsr),
    )


for root, branch, pop in sims_plot:

    sim_name = root.split('/')[-1]
    fname = root + '/' + branch + '/catalogue_data.h5'

    cat = Catalogue.from_file(pop, fname)

    _tr_scan = np.logical_and(cat.tracks['type'] == 'scan', cat.tracks['tracklet'])
    tracks_scan = cat.tracks[_tr_scan]

    detected = np.full((len(pop),), False, dtype=np.bool)

    if len(tracks_scan) > 0:

        detection_times = []
        for ind in range(cat.size):
            _test = np.logical_and(_tr_scan, cat.tracks['index'] == ind)
            if np.any(_test):
                print('- OID {} detected'.format(ind))
                _det_times = cat.tracks[_test]['SNRdB-t']
                detection_times.append(np.min(_det_times))
                detected[ind] = True

        detection_times = np.array(detection_times, dtype=cat._type)

        opts = {}
        opts['title'] = "Cumulative catalogue buildup [unique objects]"
        opts['xlabel'] = "Time past $t_0$ [h]"
        opts['ylabel'] = "Number of known objects"
        opts['label'] = "Catalogue"
        opts['cumulative'] = True
        opts['show'] = False
        fig, ax = dpt.hist(detection_times/3600.0, **opts)

        ax.plot([np.min(detection_times/3600.0), np.max(detection_times/3600.0)], [len(cat.population), len(cat.population)], '-r', label = 'Detectable population size')
        ax.legend(fontsize=24)

        print('{} exists: {}'.format(_plot_out + '/' +  sim_name, os.path.exists(_plot_out + '/' +  sim_name)))
        if not os.path.exists(_plot_out + '/' +  sim_name):
            os.mkdir(_plot_out + '/' +  sim_name)

        print('{} exists: {}'.format(_plot_out + '/' +  sim_name + '/' + branch, os.path.exists(_plot_out + '/' +  sim_name + '/' + branch)))
        if not os.path.exists(_plot_out + '/' +  sim_name + '/' + branch):
            os.mkdir(_plot_out + '/' +  sim_name + '/' + branch)
        
        plot_out = _plot_out + '/' +  sim_name + '/' + branch + '/'

        fig.savefig(plot_out + 'catalogue_buildup.png',bbox_inches='tight')

        base_pop = pop.copy()
        #------------------ FILTER
        pop.objs = pop.objs[detected]

        plot_title = 'Detected population: {}'.format(sim_name.replace('_', ' '))

        opts = {}
        opts['title'] = plot_title
        opts['xlabel'] = "Apogee [km]"
        opts['ylabel'] = "Count"
        opts['show'] = False

        print('Detected Population size "{}": {}'.format(pop.name, len(pop)))
        
        fig_d, ax_d = pop.plot_distribution('d', 'Diameter [$log_{10}$(m)]', logx = True)
        ax_d.set_title(plot_title, fontsize=24)

        fig_d.savefig(plot_out 
            + 'd_dist_'
            + branch
            + '.png', bbox_inches='tight')

        opts['xlabel'] = "Diameter [$log_{10}$(m)]"
        _logd = np.log10(base_pop['d'])
        _opts = copy.deepcopy(opts)
        _opts['color'] = 'r'
        _opts['bins'] = np.linspace(np.min(_logd), np.max(_logd), num = 200)
        _opts['label'] = 'Detectable MASTER 2009 population'
        fig_comp, ax_comp = dpt.hist(_logd, **_opts)

        _opts['plot'] = (fig_comp, ax_comp)
        _opts['color'] = 'b'
        _opts['label'] = 'Detected MASTER 2009 population'
        fig_comp, ax_comp = dpt.hist(np.log10(pop['d']), **_opts)
        ax_comp.legend(fontsize=24)
        fig_comp.savefig(plot_out 
            + 'd_compare_dist_'
            + branch
            + '.png', bbox_inches='tight')

        _opts = copy.deepcopy(opts)
        _opts['color'] = 'r'
        _opts['bins'] = np.linspace(np.min(_logd), np.max(_logd), num = 200)
        _opts['label'] = 'Detectable MASTER 2009 population'
        _opts['logy'] = True
        fig_comp, ax_comp = dpt.hist(_logd, **_opts)

        _opts['plot'] = (fig_comp, ax_comp)
        _opts['color'] = 'b'
        _opts['label'] = 'Detected MASTER 2009 population'
        fig_comp, ax_comp = dpt.hist(np.log10(pop['d']), **_opts)
        ax_comp.legend(fontsize=24)
        fig_comp.savefig(plot_out 
            + 'd_log_compare_dist_'
            + branch
            + '.png', bbox_inches='tight')


        fig_ae, ax_ae = pop.plot_distribution(
            ['a', 'e'], 
            ['Semi-major axis [log$_{10}$(km)]', 'Eccentricity [1]'],
            logx = True,
            logy = False,
            log_freq = True,
        )
        ax_ae.set_title('{}: {} / {} Objects detected'.format(sim_name.replace('_', ' '), len(pop), len(base_pop)), fontsize=24)
        fig_ae.savefig(plot_out 
            + 'ae_dist_'
            + branch
            + '.png', bbox_inches='tight')


        opts['xlabel'] = "Apogee [km]"
        _Q = pop['a']*(1.0 + pop['e'])
        fig, ax = dpt.hist(_Q[_Q < 10000.0], **opts)
        fig.savefig(plot_out 
            + 'trunc_Q_dist_'
            + branch
            + '.png', bbox_inches='tight')

        opts['xlabel'] = "Perigee [km]"
        _q = pop['a']*(1.0 - pop['e'])
        fig, ax = dpt.hist(_q[_q < 8500.0], **opts)
        fig.savefig(plot_out 
            + 'trunc_q_dist_'
            + branch
            + '.png', bbox_inches='tight')

        fig, ax = dpt.hist(_Q, **opts)
        fig.savefig(plot_out 
            + 'Q_dist_'
            + branch
            + '.png', bbox_inches='tight')

        opts['xlabel'] = "Perigee [km]"
        fig, ax = dpt.hist(_q, **opts)
        fig.savefig(plot_out 
            + 'q_dist_'
            + branch
            + '.png', bbox_inches='tight')


        opts['xlabel'] = "Inclination [deg]"
        fig, ax = dpt.hist(pop['i'], **opts)
        fig.savefig(plot_out 
            + 'inc_dist_'
            + branch
            + '.png', bbox_inches='tight')

        opts = {}
        opts['title'] = plot_title
        opts['ylabel'] = "Count"
        opts['show'] = False

        _opts = copy.deepcopy(opts)
        _opts['color'] = 'r'
        _opts['label'] = 'Detectable MASTER 2009 population'
        _opts['xlabel'] = "Inclination [deg]"
        fig, ax = dpt.hist(base_pop['i'], **_opts)

        _opts['plot'] = (fig, ax)
        _opts['color'] = 'b'
        _opts['label'] = 'Detected MASTER 2009 population'
        fig, ax = dpt.hist(pop['i'], **_opts)
        ax.legend(fontsize=18)
        fig.savefig(plot_out 
            + 'inc_comp_dist_'
            + branch
            + '.png', bbox_inches='tight')
