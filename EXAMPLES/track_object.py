#!/usr/bin/env python
#
#
import numpy as n
from mpi4py import MPI
import sys

sys.path.insert(0, "/home/danielk/IRF/IRF_GITLAB/SORTSpp")
# SORTS imports CORE
import population as p

#SORTS Libraries
import radar_library as rl
import radar_scan_library as rslib
import scheduler_library as sch
import antenna_library as alib
import space_object as so
import simulate_tracking

#initialize the radar setup
radar = rl.eiscat_3d()

radar.set_FOV(max_on_axis=25.0, horizon_elevation=30.0)
radar.set_SNR_limits(min_total_SNRdb=10.0, min_pair_SNRdb=0.0)
radar.set_TX_bandwith(bw = 1.0e6)
radar.set_beam('TX', alib.e3d_array_beam_stage1(opt='dense') )
radar.set_beam('RX', alib.e3d_array_beam() )

o = so.space_object(
    a=7000, e=0.0, i=72,
    raan=0, aop=0, mu0=0,
    C_D=2.3, A=1.0, diam=0.7,
    m=1.0,
)

#Get detections for 1 d
det_times = simulate_tracking.get_passes(o, radar, 0, 24.*3600., max_dpos=50.0)

simulate_tracking.print_passes(det_times)