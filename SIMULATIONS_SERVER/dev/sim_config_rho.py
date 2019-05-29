#!/usr/bin/env python
#
#
import numpy as n
from mpi4py import MPI
import sys

sys.path.insert(0, "/home/danielk/IRF/IRF_GITLAB/SORTSpp")
# SORTS imports
import population as p
import radar_library as rl
import radar_scan_library as rslib
import simulation as s
import scheduler_library as sch

sim_root = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/sim_rho'

#initialize the radar setup
e3d = rl.eiscat_uhf()
e3d._min_on_axis=25.0
e3d._min_SNRdb=1.0

e3d._tx[0].enr_thresh = 1.0 #SNR?
e3d._tx[0].el_thresh = 30.0 #deg
for rx in e3d._rx:
	rx.el_thresh =  30.0 #deg
e3d._tx[0].ipp = 10e-3 # pulse spacing
e3d._tx[0].n_ipp = 5 # number of ipps to coherently integrate
e3d._tx[0].pulse_length = 1e-3

#initialize the observing mode
e3d_scan = rslib.beampark_model(az=0.0, el=90.0, alt = 150, dwell_time = 0.1)

e3d_scan.set_radar_location(e3d)
e3d._tx[0].scan = e3d_scan

def pdf(a,e,i,omega,Omega,mu,s):
	pass

#load the input population
pop = p.MC_sample(pdf,num=1e5)

sim = s.simulation( \
	radar = e3d,\
	population = pop,\
	sim_root = sim_root,\
	simulation_name = s.auto_sim_name('BEAMPARK_RHO')
)

sim._verbose = True


sim.run_scan(48.0)

orbs = sim.detected_orbits(numpy=True,fname='orbs.txt')
