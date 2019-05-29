#!/usr/bin/env python
import os
import sys
import copy

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/danielk/PYTHON/SORTSpp")

import population_library as plib 
import radar_library as rlib
from propagator_orekit import PropagatorOrekit
from sorts_config import p as default_propagator
import dpt_tools as dpt
detects = [
    ('/ZFS_DATA/SORTSpp/FP_sims/TSR_beampark/MASTER2009_SGP4_500kW_1.2GHz.h5', 'Tromso Space Radar 1.2 GHz at 500 kW', 'TSR_500kw_1200Mhz')
]

plot_out = '/ZFS_DATA/SORTSpp/FP_sims/plots/detectable_population/'
master_in = "./master/celn_20090501_00.sim"

base_pop = plib.master_catalog_factor(
    input_file = master_in,
    treshhold = 0.01,
    seed=65487945,
)

for ofname, title, fname in detects:
    if os.path.exists(ofname):
        print('Output file: {}'.format(ofname))

        pop = plib.filtered_master_catalog_factor(
            radar = None,
            detectability_file = ofname,
            treshhold = 0.01,
            min_inc = 50,
            prop_time=48.0,
        )

        plot_title = 'Detectable population: {}'.format(title)

        opts = {}
        opts['title'] = plot_title
        opts['xlabel'] = "Apogee [km]"
        opts['ylabel'] = "Count"
        opts['show'] = False

        print('Population size "{}": {}'.format(pop.name, len(pop)))
        
        fig_d, ax_d = pop.plot_distribution('d', 'Diameter [$log_{10}$(m)]', logx = True)
        ax_d.set_title(plot_title, fontsize=24)

        fig_d.savefig(plot_out 
            + 'd_dist_'
            + fname
            + '.png', bbox_inches='tight')

        opts['xlabel'] = "Diameter [$log_{10}$(m)]"
        _logd = np.log10(base_pop['d'])
        _opts = copy.deepcopy(opts)
        _opts['color'] = 'r'
        _opts['bins'] = np.linspace(np.min(_logd), np.max(_logd), num = 200)
        _opts['label'] = 'MASTER 2009 population'
        fig_comp, ax_comp = dpt.hist(_logd, **_opts)

        _opts['plot'] = (fig_comp, ax_comp)
        _opts['color'] = 'b'
        _opts['label'] = 'Detectable MASTER 2009 population'
        fig_comp, ax_comp = dpt.hist(np.log10(pop['d']), **_opts)
        ax_comp.legend(fontsize=24)
        fig_comp.savefig(plot_out 
            + 'd_compare_dist_'
            + fname
            + '.png', bbox_inches='tight')

        _opts = copy.deepcopy(opts)
        _opts['color'] = 'r'
        _opts['bins'] = np.linspace(np.min(_logd), np.max(_logd), num = 200)
        _opts['label'] = 'MASTER 2009 population'
        _opts['logy'] = True
        fig_comp, ax_comp = dpt.hist(_logd, **_opts)

        _opts['plot'] = (fig_comp, ax_comp)
        _opts['color'] = 'b'
        _opts['label'] = 'Detectable MASTER 2009 population'
        fig_comp, ax_comp = dpt.hist(np.log10(pop['d']), **_opts)
        ax_comp.legend(fontsize=24)
        fig_comp.savefig(plot_out 
            + 'd_log_compare_dist_'
            + fname
            + '.png', bbox_inches='tight')


        fig_ae, ax_ae = pop.plot_distribution(
            ['a', 'e'], 
            ['Semi-major axis [log$_{10}$(km)]', 'Eccentricity [1]'],
            logx = True,
            logy = False,
            log_freq = True,
        )
        ax_ae.set_title('{}: Detectable {} Objects'.format(title,  len(pop)), fontsize=24)
        fig_ae.savefig(plot_out 
            + 'ae_dist_'
            + fname
            + '.png', bbox_inches='tight')


        opts['xlabel'] = "Apogee [km]"
        _Q = pop['a']*(1.0 + pop['e'])
        fig, ax = dpt.hist(_Q[_Q < 10000.0], **opts)
        fig.savefig(plot_out 
            + 'trunc_Q_dist_'
            + fname
            + '.png', bbox_inches='tight')

        opts['xlabel'] = "Perigee [km]"
        _q = pop['a']*(1.0 - pop['e'])
        fig, ax = dpt.hist(_q[_q < 8500.0], **opts)
        fig.savefig(plot_out 
            + 'trunc_q_dist_'
            + fname
            + '.png', bbox_inches='tight')

        fig, ax = dpt.hist(_Q, **opts)
        fig.savefig(plot_out 
            + 'Q_dist_'
            + fname
            + '.png', bbox_inches='tight')

        opts['xlabel'] = "Perigee [km]"
        fig, ax = dpt.hist(_q, **opts)
        fig.savefig(plot_out 
            + 'q_dist_'
            + fname
            + '.png', bbox_inches='tight')


        opts['xlabel'] = "Inclination [deg]"
        fig, ax = dpt.hist(pop['i'], **opts)
        fig.savefig(plot_out 
            + 'inc_dist_'
            + fname
            + '.png', bbox_inches='tight')


#plt.show()
