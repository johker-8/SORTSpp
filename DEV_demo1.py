#Project 1: Create a funky spiral scan!

import numpy as np
import matplotlib.pyplot as plt

#SORTS++
import radar_scans
import radar_scan_library

#step sizes
d_az = 3.0
d_el = 0.1
az = [0.0]
el = [90.0]
while el[-1] > 50.0:
    az += [ np.mod(az[-1] + d_az, 360.0) ]
    el += [ el[-1] - d_el ]

SC = radar_scan_library.n_const_pointing_model(az, el, 69, 31, 150, dwell_time=0.1)

#Now we dont want to plt
#radar_scans.plot_radar_scan(SC, earth=True)
#plt.show()