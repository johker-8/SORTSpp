#!/usr/bin/env python
#
#
import orbital_estimation as oe

tracklet_folder = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/sim_1/tracklets/12303250'
prior_file = '/home/danielk/IRF/E3D_PA/SORTSpp_sim/sim_1/prior/12303250.oem'

ecef_list  = oe.state_estimation(tracklet_folder, verbose = True)

oe.write_ecef_to_oem(ecef_list[0],prior_file)