import numpy as n
import scipy.constants as c

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap


def plot_radar_conf(radar, save=False):
    '''Plots the geographical situation of an instance of :class:`radar_config.RadarSystem`.
    
    # TODO: Make sure this function is up to date and general, also look over the Earth plotting.
    '''
    
    for r in radar._tx:
      rs.plot_radar_scan(r.scan)

    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    map = Basemap(projection = 'aeqd',
              width = 0.7e6, height = 0.7e6, resolution = "l", ax = ax,
                  lat_0 = radar._tx[0].lat, lon_0 = radar._tx[0].lon)
    
    map.drawmapboundary(fill_color = 'white', color = "gray")
    map.fillcontinents(color = 'white', lake_color = 'white')
    map.drawcoastlines(color = "gray")
    
    map.drawparallels(range(0, 90, 2), labels = [1, 1, 0, 0])
    map.drawmeridians(range(0, 360, 5), labels = [0, 0, 1, 1])
    
    for r in radar._rx:
        x, y = map(r.lon, r.lat)
        map.plot(x, y, 'ro', markersize = 15, label = "RX")
        print(r.name)
        print(x)
        print(y)
        plt.text(x-50000, y+30000, r.name)
    print("done")
    for r in radar._tx:
        x, y = map(r.lon, r.lat)
        map.plot(x, y, 'w*', markersize = 15, label = "TX")
        plt.text(x-50000, y+30000, r.name)

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(numpoints = 1)
    if save:
      plt.savefig(save)
    plt.show()
