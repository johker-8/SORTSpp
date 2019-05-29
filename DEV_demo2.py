#Project 2: Lets put an EISCAT 3D module in my home island!

import radar_library
import radar_config

import matplotlib.pyplot as plt

radar = radar_library.eiscat_3d_module()

radar._tx[0].lat = 59.993693
radar._tx[0].lon = 20.166953

radar._rx[0].lat = 59.993693
radar._rx[0].lon = 20.166953

radar._rx[1].lat = 60.008134
radar._rx[1].lon = 18.566308
radar._rx[2].lat = 60.555240
radar._rx[2].lon = 21.428962

#Now we dont want to plt
#radar_config.plot_radar(radar)
#plt.show()