'''
This script file is meant to be run using the python interpeter and the -i option. It will load the nessesery pacages and provide a few custom functions useful for examining the output data of SORTS++ manually.

Example:

   fig, ax = dpt.hist
'''
import os

import h5py
import numpy as n
import matplotlib.pyplot as plt
import dpt_tools as dpt
from glob import glob

def get_raw_paths(dir, open_id, verbose = False):
    lst_objs = glob(sim_root+'/tracklets/*/')

    #filter out only object names
    #this works since string is ..../ID/
    #then split will create [...., ID,'']
    #so the next to last element [-2] is the id
    objs = [ x.split('/')[-2] for x in lst_objs]

    #first 4 are for factor ID
    master_objs = [x[4:] for x in objs]
    factor_objs = [x[:4] for x in objs]

    tracklets = glob(lst_objs[open_id]+'*.tdm')

    tracklet_data = [tr.split('-') for tr in tracklets]
    #format: time, master ID (including factor), tx_rx index
    tracklet_data = [(tr[1],tr[2],tr[3][:3]) for tr in tracklet_data]

    if verbose:
        print('Tracklets for folder {}:'.format(lst_objs[open_id]))
        for i,tr in enumerate(tracklet_data):
            print('\n-- PATH: {}'.format(tracklets[i]) )
            print('Master ID: {}'.format(tr[1][4:]))
            print('Factor ID: {}'.format(tr[1][:4]) )
            print('TX: {} to RX: {}'.format(tr[2][0], tr[2][2]))
            print('Unix time: {}'.format(tr[0]) )

        print('-------------------')

    lst_priors = glob(sim_root+'/prior/*_init.oem')

    #here we dont have trailing /
    priors = [ x.split('/')[-1] for x in lst_priors]

    #remove end of file name
    prior_id = [ x.split('_')[0] for x in priors]

    #first 4 are factor ID
    master_prior = [x[4:] for x in prior_id]
    factor_prior = [x[:4] for x in prior_id]

    if verbose:
        print('Path to prior {}: {}'.format(open_id, lst_priors[open_id]))
        print('With ID: {}'.format(prior_id[open_id]))
        print('Master ID: {}, factor ID: {}'.format(master_prior[open_id], factor_prior[open_id]))

def open_meta(ind):
    return h5py.File(root + meta_files[ind],'r')

use_version = raw_input('Do your meta files have a version? [y/n]:')

if use_version.lower() == 'y':
    version = raw_input('Please state version: ')
else:
    version = ''



meta_files = [
    'scheduled_detections{}.h5'.format(version),
    'scheduled_tracks{}.h5'.format(version),
    'scheduled_maint{}.h5'.format(version),
    'detections/schedule.h5',
    'detections/scan_meta.h5',
]

use_root = raw_input('Do you want to give a root to a simulation? [y/n]: ')


if use_root.lower() == 'y':
    root = raw_input('Please state root: ')
    if root[-1] != '/':
        root+='/'
    if not os.path.exists(root):
        raise Exception('Given root folder does not exist')
else:
    root = ''


from pylab import *
ion()
