#!/usr/bin/env python

'''Orbit determination tests.

'''

#
#    Standard python
#
import os
import copy
import glob

#
#   External packages
#
import h5py
from mpi4py import MPI
from pandas.plotting import scatter_matrix
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

comm = MPI.COMM_WORLD

#
#   SORTS ++
#
import space_object
import dpt_tools as dpt
import TLE_tools as tle
import plothelp
import sources
import orbit_determination
import population_library as plib
from sorts_config import p as default_propagator

def gen_tracklet_prior(fout, index = 10):
    pop = plib.master_catalog()
    space_o = pop.get_object(index)



def try_plots(object_id = 8400199):
    results = orbit_determination.Parameters.load_h5('./tests/tmp_test_data/test_sim/master/orbits/8400199_orbit_determination.h5')

    obs_dates = None
    with h5py.File('./tests/tmp_test_data/test_sim/master/orbits/8400199_obs_data.h5') as hf:
        for key in hf:
            if obs_dates is None:
                obs_dates = [dpt.mjd2npdt(hf['{}/mjd'.format(key)].value)]
            else:
                obs_dates += [dpt.mjd2npdt(hf['{}/mjd'.format(key)].value)]

    pop = plib.master_catalog()
    pop_ind = np.argwhere(pop['oid'] == int(object_id))[0,0]
    space_o = pop.get_object(pop_ind)

    orbit_determination.plot_orbit_uncertainty(results, 
        true_object = space_o, 
        time = 24.0,
        obs_dates = obs_dates, 
        num=len(results.trace),
        symmetric = True,
        )
    plt.show()
    exit()

    orbit_determination.plot_trace(results)
    orbit_determination.plot_MC_cov(results)
    orbit_determination.plot_scatter_trace(results)
    orbit_determination.plot_autocorrelation(results)
    orbit_determination.print_covariance(results)

    plt.show()


def try_od(object_id = 8400199):

    #object_id = 121800762

    pop = plib.master_catalog()
    pop_ind = np.argwhere(pop['oid'] == int(object_id))[0,0]
    space_o = pop.get_object(pop_ind)

    res = orbit_determination.orbit_determination_fsystem(
        path='./tests/tmp_test_data/test_sim/master/',
        variables=['x', 'y', 'z', 'vx', 'vy', 'vz', 'A'],
        params={
            'A': 1.0,
        },
        object_id = object_id,
        true_object = space_o,
        prior_truncate = slice(None,None,50),
        #tracklet_truncate = slice(None,None,50),
        steps = 3000,
        tune = 1000,
        output_folder = './tests/tmp_test_data/test_sim/master/orbits',
    )

    if comm.rank != 0:
        exit()

    orbit_determination.plot_orbits(res['mcmc'], start=res['start'])

    results = res['results']

    orbit_determination.plot_trace(results)
    orbit_determination.plot_MC_cov(results)
    orbit_determination.plot_scatter_trace(results)
    orbit_determination.plot_autocorrelation(results)
    orbit_determination.print_covariance(results)

    styles = ['-r','-b','-g','-k']
    labels = ['MAP MCMC-SCAM', 'MAP Nelder-Mead', 'Estimated Prior', 'Start value']
    states = [res['mcmc'].results.MAP, res['MAP-estimation'].results.MAP, res['prior'].results.MAP, res['start']]
    orbit_determination.plot_residuals(res['mcmc'], states, labels, styles)

    plt.show()

    

def gen_simulation(part = 4):

    import time
    import sys
    import os
    import logging
    import glob
    import shutil
    import pymc3 as pm
    
    import h5py
    import scipy
    from mpi4py import MPI
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # SORTS imports CORE
    import population_library as plib
    from simulation import Simulation
    from population import Population
    
    #SORTS Libraries
    import radar_library as rlib
    import population_library as plib
    import radar_scan_library as rslib
    import scheduler_library as schlib
    import antenna_library as alib
    import rewardf_library as rflib
    import plothelp
    import dpt_tools as dpt
    import ccsds_write


    sim_root =  './tests/tmp_test_data/test_sim2'

    np.random.seed(12345) #NEEDED 

    SIM_TIME = 24.0*4

    pop = plib.master_catalog()
    pop.objs = pop.objs[:10]

    #initialize the radar setup
    radar = rlib.eiscat_3d(beam='interp', stage=1)

    radar.set_FOV(max_on_axis=90.0, horizon_elevation=30.0)
    radar.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=1.0)
    radar.set_TX_bandwith(bw = 1.0e6)

    sim = Simulation(
        radar = radar,
        population = pop,
        root = sim_root,
        scheduler = schlib.dynamic_scheduler,
        simulation_name = 'OD validation',
    )

    #sim.set_log_level(logging.DEBUG)

    sim.observation_parameters(
        duty_cycle=0.25,
        SST_fraction=1.0,
        tracking_fraction=1.0,
        SST_time_slice=0.2,
    )
    sim.simulation_parameters(
        tracklet_noise=True,
        max_dpos=50e3,
        auto_synchronize=True,
    )

    #We know all!
    sim.catalogue.maintain(slice(None))

    ################## RUNNING #####################

    if part == 1:
        sim.run_observation(SIM_TIME)

        sim.status(fout='{}h_to_{}h'.format(0, int(SIM_TIME)))

        sim.print_maintenance()
        sim.print_detections()

        sim.set_scheduler_args(
            reward_function = rflib.rewardf_exp_peak_SNR_tracklet_len,
            reward_function_config = {
                'sigma_t': 60.0*5.0,
                'lambda_N': 50.0,
            },
            logger = sim.logger,
        )

        sim.run_scheduler()
        sim.print_tracks()
        sim.print_tracklets()

        sim.status(fout='{}h_to_{}h_scheduler'.format(0, int(SIM_TIME)))
    
    if part == 2:
        sim.load()
        sim.generate_tracklets()
    
    if part == 3:
        sim.load()
        sim.generate_priors(
            frame_transformation='TEME',
            tracklet_truncate = slice(None, None, 20),
        )

    if part == 4:
        sim.load()
        sim.run_orbit_determination(
            frame_transformation = 'TEME',
            error_samp = 500,
            steps = 10000, 
            max_zenith_error = 0.9,
            tracklet_truncate = slice(None, None, 400),
            tune = 1000,
        )



if __name__=='__main__':
    gen_simulation(part=4)

    #try_plots()
    #try_od()
    