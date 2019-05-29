#!/usr/bin/env python
#
# plot eiscat 2018 beampark pointing directions
#
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

import coord
import plothelp

import radar_scans as rs
import radar_scan_library as rslib

uhf = rslib.beampark_model(90.0,75.0,lat=69.58,lon=19.23)
esr = rslib.beampark_model(90.0,75.0,lat=78.15,lon=16.02)

scans.append(uhf)
scans.append(esr)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(15, 5)
plothelp.draw_earth_grid(ax)

plothelp.draw_radar(ax,69.58,19.23,name="UHF",color="red")
plothelp.draw_radar(ax,78.15,16.02,name="ESR",color="blue")

p0,k0=uhf.antenna_pointing(0.0)
p1=p0+k0*3000e3
ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="green")

p0,k0=esr.antenna_pointing(0.0)
p1=p0+k0*3000e3
ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="green")

plt.title("2018 EISCAT Beampark pointings")
plt.legend()
plt.tight_layout()

plt.show()
