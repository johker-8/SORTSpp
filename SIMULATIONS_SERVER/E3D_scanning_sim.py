#!/usr/bin/env python
#
#
import numpy as np
from mpi4py import MPI
import sys

# replace this with the path to your sorts
sys.path.insert(0, "/home/danielk/PYTHON/SORTSpp")

# SORTS imports CORE
import population_library as plib
from simulation import Simulation

#SORTS Libraries
import radar_library as rlib
import radar_scan_library as rslib
import scheduler_library as schlib
import antenna_library as alib
import rewardf_library as rflib


####### RUN CONFIG #######
part = 11
SIM_TIME = 24.0*1.0
##########################

sim_root = '/ZFS_DATA/SORTSpp/FP_sims/E3D_scanning'

#initialize the radar setup 
radar = rlib.eiscat_3d(beam='interp', stage=1)

radar.set_FOV(max_on_axis=90.0, horizon_elevation=30.0)
radar.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=1.0)
radar.set_TX_bandwith(bw = 1.0e6)

scan = rslib.ns_fence_rng_model(
    radar._tx[0].lat,
    radar._tx[0].lon,
    radar._tx[0].alt, 
    min_el = 30, 
    angle_step = 2.0, 
    dwell_time = 0.2,
)

radar.set_scan(scan)

#load the input population
pop = plib.filtered_master_catalog_factor(
    radar = radar,
    treshhold = 0.01,
    min_inc = 50,
    prop_time = 48.0,
)

sim = Simulation(
    radar = radar,
    population = pop,
    root = sim_root,
    scheduler = schlib.dynamic_scheduler,
    simulation_name = 'E3D scanning of MASTER2009'
)

sim.observation_parameters(
    duty_cycle=0.25,
    SST_fraction=1.0,
    tracking_fraction=0.25,
    SST_time_slice=0.2,
)
sim.simulation_parameters(
    tracklet_noise=True,
    max_dpos=50e3,
    auto_synchronize=True,
)


################## RUNNING #####################

if part == 1:
    sim.run_observation(SIM_TIME)

    sim.status(fout='{}h_to_{}h'.format(0, int(SIM_TIME)))

    sim.print_maintenance()
    sim.print_detections()


if part == 2.2:
    sim.set_version('scheduler_2d')
    sim.checkout_simulation('master') #load included

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

if part == 3.2:
    sim.set_version('scheduler_2d')
    sim.load()
    sim.plots()

if part == 4.2:
    sim.set_version('scheduler_2d')
    sim.load()
    
    sim.generate_tracklets()
    sim.generate_priors()


if part == 11:
    sim.set_version('tight_fence')

    beam_width = 2.0 #deg
    max_sat_speed = 8e3 #m/s
    leak_proof_at = 150e3 #range
    beam_arc_len = np.radians(beam_width)*leak_proof_at
    sat_traverse_time = beam_arc_len/max_sat_speed #also max scan time

    els = ['90.0', '88.73987398739877', '87.4357183693167', '86.1720061939742', '84.90239189256025', '83.628529014365', '82.35204237515919', '81.07452478944677', '79.77710211752762', '78.47281161299796', '77.16391480976668', '75.83375939658995', '74.49527977424731', '73.13358804398229', '71.753175185289', '70.3584796645483', '68.93771910430019', '67.48130276816576', '65.99689473774336', '64.48487395668005', '62.93289943117691', '61.33878768753357', '59.69647098103616', '57.99766296011879', '56.23923388331565', '54.40755219009088', '52.44010673902446', '50.4247836215483', '48.29222295539205', '46.01278403125778', '43.55617730018973', '40.8717857555977', '37.886094818016815', '34.478961143266076', '30.0']
    els = [float(el) for el in els]
    azs = [-180.0]*(len(els)-1) + [0.0]*len(els)
    els = els[1:][::-1] + els

    dwell_time = sat_traverse_time/float(len(azs))

    scan = rslib.n_const_pointing_model(
        az = azs,
        el = els,
        lat = radar._tx[0].lat,
        lon = radar._tx[0].lon,
        alt = radar._tx[0].alt,
        dwell_time = dwell_time,
    )

    sim.radar.set_scan(scan)

    sim.observation_parameters(
        duty_cycle=0.25,
        SST_fraction=1.0,
        tracking_fraction=0.25,
        SST_time_slice=dwell_time,
    )

    sim.run_observation(SIM_TIME)

    sim.status(fout='{}h_to_{}h'.format(0, int(SIM_TIME)))

    sim.print_maintenance()
    sim.print_detections()

if part == 12:
    sim.set_version('tight_fence_schedule')
    sim.checkout_simulation('tight_fence') #load included

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

if part == 13:
    sim.set_version('tight_fence_schedule')
    sim.load()
    sim.plots()