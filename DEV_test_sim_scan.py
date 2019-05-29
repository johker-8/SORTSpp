
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import space_object
import radar_library
import simulate_scan
import plothelp
import radar_scans



def plot_detections(radar, space_o, t_end=24.0*3600.0):

    t = np.linspace(0, t_end, num=10000)

    detections = simulate_scan.get_detections(space_o, radar, 0.0, t_end)
    simulate_scan.pp_det(detections)

    detections = detections[0]

    passes = np.unique(np.array(detections['t0']))

    print('{} detections of object'.format(len(detections['tm'])))
    print('detection on {} passes'.format(len(passes)))

    ecefs1 = space_o.get_state(t)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)

    radar_scans.plot_radar_scan(radar._tx[0].scan, earth=True, ax=ax)


    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],"-",color="green", alpha=0.5)

    for det in detections['tm']:
        ecefs2 = space_o.get_state([det])
        ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],".",color="red", alpha=0.5)
        ax.plot(
            [radar._tx[0].ecef[0], ecefs2[0,0]], 
            [radar._tx[0].ecef[1], ecefs2[1,0]], 
            [radar._tx[0].ecef[2], ecefs2[2,0]],
            "-",color="red", alpha=0.1,
        )


    box = 1000e3

    ax.set_xlim([radar._tx[0].ecef[0] - box, radar._tx[0].ecef[0] + box])
    ax.set_ylim([radar._tx[0].ecef[1] - box, radar._tx[0].ecef[1] + box])
    ax.set_zlim([radar._tx[0].ecef[2] - box, radar._tx[0].ecef[2] + box])



space_o = space_object.SpaceObject(
    a=7000, e=0.0, i=69,
    raan=0, aop=0, mu0=0,
    C_D=2.3, A=1.0, m=1.0,
    C_R=1.0, oid=42, d=0.1,
    mjd0=57125.7729,
)

print(space_o)

radars = [
    radar_library.eiscat_uhf(),
    radar_library.eiscat_3d(),
]

radars[0]._tx[0].scan.keyword_arguments(el=45.0)

for radar in radars:
    plot_detections(radar, space_o, t_end=24.0*3600.0)

plt.show()

