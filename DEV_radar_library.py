import numpy as np
import scipy.constants as c

import matplotlib.pyplot as plt

from radar_config import plot_radar
import radar_library as rl

radars = [
    rl.eiscat_3d(beam = 'gauss'),
    #rl.eiscat_3d(beam = 'array'),
    rl.eiscat_3d_module(beam = 'gauss'),
    #rl.eiscat_3d_module(beam = 'array'),
    rl.eiscat_svalbard(),
    rl.eiscat_uhf(),
]

for radar in radars:
    plot_radar(radar)



