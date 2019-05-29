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
sys.path.insert(0, "/home/danielk/IRF/IRF_GITLAB/SORTSpp")

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

_plot_out = '/home/danielk/IRF/E3D_PA/FP_sims/plots/detection_rates'


sims_plot = [
    ('/home/danielk/IRF/E3D_PA/FP_sims/TSR_fence_beampark/MASTER2009_SGP4_500kW_1.2GHz', 0.001),
    ('/home/danielk/IRF/E3D_PA/FP_sims/TSR_beampark/MASTER2009_SGP4_100kW_1.2GHz', 0.001),
    ('/home/danielk/IRF/E3D_PA/FP_sims/TSR_beampark/MASTER2009_SGP4_500kW_1.2GHz', 0.01),
]

for sim_path, th in sims_plot:

    branch = sim_path.split('/')[-1]
    sim_name = sim_path.split('/')[-2]
    sim_root = '/'.join(sim_path.split('/')[:-1])
    fname = sim_root + '/' + branch + '/catalogue_data.h5'

    pop = plib.filtered_master_catalog_factor(
        radar = None,
        treshhold = th,
        min_inc = 50,
        prop_time = 48.0,
        detectability_file = sim_root + '/' + branch + '.h5',
    )

    cat = Catalogue.from_file(pop, fname)

    _tr_scan = np.logical_and(cat.tracks['type'] == 'scan', cat.tracks['tracklet'])
    tracks_scan = cat.tracks[_tr_scan]

    detected = np.full((len(pop),), False, dtype=np.bool)

    if len(tracks_scan) > 0:

        detection_rates = cat.tracks[_tr_scan]['t0']

        detection_times = []
        for ind in range(cat.size):
            _test = np.logical_and(_tr_scan, cat.tracks['index'] == ind)
            if np.any(_test):
                #print('- OID {} detected'.format(ind))
                _det_times = cat.tracks[_test]['SNRdB-t']
                detection_times.append(np.min(_det_times))
                detected[ind] = True

        detection_times = np.array(detection_times, dtype=cat._type)

        print('{} exists: {}'.format(_plot_out + '/' +  sim_name, os.path.exists(_plot_out + '/' +  sim_name)))
        if not os.path.exists(_plot_out + '/' +  sim_name):
            os.mkdir(_plot_out + '/' +  sim_name)

        print('{} exists: {}'.format(_plot_out + '/' +  sim_name + '/' + branch, os.path.exists(_plot_out + '/' +  sim_name + '/' + branch)))
        if not os.path.exists(_plot_out + '/' +  sim_name + '/' + branch):
            os.mkdir(_plot_out + '/' +  sim_name + '/' + branch)
        
        plot_out = _plot_out + '/' +  sim_name + '/' + branch + '/'

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

        fig.savefig(plot_out + 'catalogue_buildup.png',bbox_inches='tight')


        opts = {}
        opts['title'] = "Simulated detection rates [non-unique]"
        opts['xlabel'] = "Time past $t_0$ [h]"
        opts['ylabel'] = "Detections per hour"
        opts['bin_size'] = 1.0
        opts['cumulative'] = False
        opts['show'] = False
        fig, ax = dpt.hist(detection_rates/3600.0, **opts)

        fig.savefig(plot_out + 'detection_rates.png',bbox_inches='tight')

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
