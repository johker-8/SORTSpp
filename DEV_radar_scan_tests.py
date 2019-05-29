#!/usr/bin/env python
#
#
#test radar pointing class
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

import coord
import plothelp

import radar_scans as rs
import radar_scan_library as rslib

#this function needs
#dwell time from class
#az el
def point_double_beampark(data,t):
	n = np.floor(t/data['dwell_time'])
	n = int(n % 2)
	return (data['az'][n],data['el'][n])

def point_ew_fence_scan(data,t):
	n = np.floor(t/data['dwell_time'] % len(data['angles']))
	angle=data['angles'][int(n)]
	e=np.cos(np.pi*angle/180.0)
	d=-np.sin(np.pi*angle/180.0)
	return (0.0,e,d)

def point_ns_fence_scan(data,t):
	n = np.floor(t/data['dwell_time'] % len(data['angles']))
	angle=data['angles'][int(n)]
	no=np.cos(np.pi*angle/180.0)
	d=-np.sin(np.pi*angle/180.0)
	return (no,0.0,d)

def point_cross_fence_scan(data,t):
	n_ang = len(data['angles'])
	n = np.floor(t/data['dwell_time'] % n_ang)
	angle=data['angles'][int(n)]

	if n <= np.round(n_ang*0.5):
		no=np.cos(np.pi*angle/180.0)
		d=-np.sin(np.pi*angle/180.0)
		e = 0.0
	else:
		e=np.cos(np.pi*angle/180.0)
		d=-np.sin(np.pi*angle/180.0)
		no = 0.0
	return (no,e,d)

def point_circle_fence(data,t):
	n = int(np.floor(t/data['dwell_time'] % data['n']))
	return (data['az'][n],data['el'][n])



def calculate_fence_angles(min_el,angle_step,dwell_time):
	angular_extent=180.0-2*min_el
	n_pos=int(angular_extent/angle_step)
	angles=np.arange(n_pos)*angle_step+min_el
	return angles




SCAN1 = rs.radar_scan(lat = 69,lon = 19,alt = 150,\
	pointing_function = point_double_beampark, \
	name = 'Double beampark scan', \
	pointing_coord = 'azel')
SCAN1._function_data['az'] = [120,240]
SCAN1._function_data['el'] = [77.5,77.5]
SCAN1._scan_time = 2*SCAN1._function_data['dwell_time']

SCANS = [SCAN1]

SCAN2 = rs.radar_scan(lat = 69,lon = 19,alt = 150,\
	pointing_function = point_ew_fence_scan, \
	name = 'East west fence scan', \
	pointing_coord = 'ned')
SCAN2._function_data['angles'] = calculate_fence_angles(min_el = 30, angle_step = 1, dwell_time = SCAN2._function_data['dwell_time'])
SCAN2._scan_time = len(SCAN2._function_data['angles'])*SCAN2._function_data['dwell_time']

SCANS.append(SCAN2)


#the copy function allows new radar scan schemas to be created very fast without risk of chainging previus scan profiles
#this also allows for a template library to be created
SCAN3 = SCAN2.copy()
SCAN3._pointing_function = point_ns_fence_scan
SCAN3._name =  'North south fence scan'

SCANS.append(SCAN3)


SCAN4 = SCAN2.copy()
SCAN4._pointing_function = point_cross_fence_scan
SCAN4._scan_time = 2*SCAN4._scan_time
SCAN4._name =  'Cross fence scan'
angs = SCAN4._function_data['angles']
SCAN4._function_data['angles'] = np.concatenate([angs,angs])

SCANS.append(SCAN4)

SCAN5 = rs.radar_scan(lat = 69,lon = 19,alt = 150,\
	pointing_function = point_circle_fence, \
	name = 'Circle fence scan', \
	pointing_coord = 'azel')
n_scan5 = 100
SCAN5._function_data['n'] = n_scan5
SCAN5._function_data['az'] = np.linspace(0,360,num=n_scan5)
SCAN5._function_data['el'] = [50 for i in range(n_scan5)]
SCAN5._scan_time = n_scan5*SCAN5._function_data['dwell_time']

SCANS.append(SCAN5)


SCAN6 = rslib.beampark_model(69,19)
SCANS.append(SCAN6)

for SC in SCANS:
	t=np.linspace(0,SC._scan_time,num=np.round(2*SC._scan_time/SC._function_data['dwell_time']))

	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111, projection='3d')
	ax.view_init(15, 5)
	plothelp.draw_earth_grid(ax)
	plothelp.draw_radar(ax,69,19)    
	for i in range(len(t)):
		p0,k0=SC.antenna_pointing(t[i])
		p1=p0+k0*3000e3
		ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="green")
		
	plt.title(SC.name())
	plt.tight_layout()

plt.show()
