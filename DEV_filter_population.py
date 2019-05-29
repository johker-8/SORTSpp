#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import population_library as plib 
import radar_library as rlib
from propagator_orekit import PropagatorOrekit
from sorts_config import p as default_propagator

radars = [
    rlib.eiscat_3d(),
    #rlib.eiscat_3d(stage = 2),
    rlib.eiscat_3d_module(),
]

master_in = "./master/celn_20090501_00.sim"

SNRs = [1.0, 10.0]

for SNR_lim in SNRs:
    for radar in radars:
        radar.set_SNR_limits(SNR_lim, SNR_lim)

        ofname = master_in[:-4]\
            + '_' + radar.name.replace(' ', '_')\
            + '_' + default_propagator.__name__\
            + '_' + str(int(np.round(radar.min_SNRdb))) + 'SNRdB'\
            + '.h5'
        print('Output file: {}'.format(ofname))
        
        pop = plib.filtered_master_catalog_factor(
            radar = radar,
            treshhold = 0.01,
            min_inc = 50,
            prop_time=48.0,
        )

        print('Population size "{}": {}'.format(pop.name, len(pop)))
        
        fig_d, ax_d = pop.plot_distribution('d', 'Diameter [$log_{10}$(m)]', logx = True)
        ax_d.set_title('48h population filter: {} '.format(ofname.replace('_',' ')), fontsize=24)

        fig_ae, ax_ae = pop.plot_distribution(
            ['a', 'e'], 
            ['Semi-major axis [log$_{10}$(km)]', 'Eccentricity [1]'],
            logx = True,
            logy = False,
            log_freq = True,
        )
        ax_ae.set_title('{}: {} Objects'.format(ofname.replace('_',' '), len(pop)), fontsize=24)


plt.show()