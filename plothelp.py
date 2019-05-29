#!/usr/bin/env python

'''Functions for making plots quicker.

'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import coord
import numpy as n

def draw_earth(ax):
    n_earth=1000
    r_e=6300e3
    earth=n.zeros([3,n_earth])
    for ei in range(n_earth):
        a0=n.random.uniform(0.0,2.0*n.pi)
        a1=n.random.uniform(0.0,n.pi)            
        earth[:,ei]=n.array([r_e*n.cos(a0)*n.sin(a1),
                             r_e*n.sin(a0)*n.sin(a1),
                             r_e*n.cos(a1)])
    ax.plot(earth[0,:],earth[1,:],earth[2,:],".",color="blue",alpha=0.1)
    # north pole
    ax.plot([0.0],[0.0],[r_e],"o",color="red")
    # south pole
    ax.plot([0.0],[0.0],[-r_e],"o",color="yellow")    

def draw_earth_grid(ax,num_lat=25,num_lon=50,alpha=0.1,res = 100, color='black'):
    lons = n.linspace(-180, 180, num_lon+1) * n.pi/180 
    lons = lons[:-1]
    lats = n.linspace(-90, 90, num_lat) * n.pi/180 

    lonsl = n.linspace(-180, 180, res) * n.pi/180 
    latsl = n.linspace(-90, 90, res) * n.pi/180 

#    r_e=6300e3
    r_e=6371e3    
    for lat in lats:
        x = r_e*n.cos(lonsl)*n.cos(lat)
        y = r_e*n.sin(lonsl)*n.cos(lat)
        z = r_e*n.ones(n.size(lonsl))*n.sin(lat)
        ax.plot(x,y,z,alpha=alpha,linestyle='-', marker='',color=color)

    for lon in lons:
        x = r_e*n.cos(lon)*n.cos(latsl)
        y = r_e*n.sin(lon)*n.cos(latsl)
        z = r_e*n.sin(latsl)
        ax.plot(x,y,z,alpha=alpha,color=color)
    
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')

def draw_radar(ax,lat,lon,name="radar",color="black"):
    n_earth=1000
    earth=coord.geodetic2ecef(lat,lon,0.0)
#    print(earth)
    ax.plot([earth[0]],[earth[1]],[earth[2]],"x",color=color,label=name)    
        
