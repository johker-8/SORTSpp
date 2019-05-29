#!/usr/bin/env python

'''A collection of :class:`radar_scans.RadarScan` instances, such as fence scans or ionospheric grids.

'''

 #test radar pointing class
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

import coord
import plothelp

import radar_scans as rs

def point_beampark(t, az, el, **kw):
    '''Pointing function for a single point beampark. AZ-EL coordinate system.
    
    :param float t: Seconds past epoch.
    :param float az: Azimuth in degrees east of north.
    :param float el: Elevation in degrees above horizon.
    '''
    return az, el, 1.0

def point_n_beampark(t, az, el, dwell_time, **kw):
    '''Pointing function for a n-point beampark with fixed dwell-times. AZ-EL coordinate system.
    
    :param float t: Seconds past epoch.
    :param list az: Azimuths in degrees east of north.
    :param list el: Elevations in degrees above horizon.
    :param float dwell_time: Time in seconds spent at each azimuth-elevation pair of pointing direction.
    '''
    ind = np.floor(t/dwell_time)
    ind = int(ind % len(az))
    return az[ind], el[ind], 1.0

def point_n_dyn_beampark(t, az, el, scan_time, dwell_time, **kw):
    '''Pointing function for a n-point beampark with variable dwell-times. AZ-EL coordinate system.
    
    :param float t: Seconds past epoch.
    :param list az: Azimuths in degrees east of north.
    :param list el: Elevations in degrees above horizon.
    :param float scan_time: Total scan time, sum of dwell times.
    :param list dwell_time: Times in seconds to spend at each azimuth-elevation pair of pointing direction.
    '''
    t = t % scan_time
    elap_t = np.cumsum(dwell_time)
    ind = np.argmax(elap_t > t)
    return az[ind], el[ind], 1.0


def point_ew_fence_scan(t, angles, dwell_time, **kw):
    '''Pointing function for a east-west fence scan. NED coordinate system.
    '''
    ind = np.floor((t/dwell_time) % len(angles))
    angle = angles[int(ind)]
    e = np.cos(np.pi*angle/180.0)
    d = -np.sin(np.pi*angle/180.0)
    return 0.0, e, d

def point_ns_fence_scan(t, angles, dwell_time, **kw):
    '''Pointing function for a north-south fence scan. NED coordinate system.
    '''
    ind = np.floor((t/dwell_time) % len(angles))
    angle = angles[int(ind)]
    no = np.cos(np.pi*angle/180.0)
    d = -np.sin(np.pi*angle/180.0)
    return no, 0.0, d

def point_sph_rng_scan(t, dwell_time, min_el, state, **kw):
    '''Pointing function for a spherically uniform random scan with a minimum elevation and fixed state at epoch, i.e reproducable sequence. NED coordinate system.
    '''
    t_n = int(np.floor(t/dwell_time))

    np.random.seed(state + t_n)
    in_FOV = False
    while not in_FOV:
        on_sph = False
        while not on_sph:
            xi = np.random.rand(4)*2.0 - 1
            xin = xi.dot(xi)
            if xin < 1.0:
                on_sph = True

        x = 2.0*(xi[1]*xi[3] + xi[0]*xi[2])/xin
        y = 2.0*(xi[2]*xi[3] - xi[0]*xi[1])/xin
        z = (xi[0]**2 + xi[3]**2 - xi[1]**2 - xi[2]**2)/xin

        if np.arcsin(z) >= min_el:
            in_FOV = True
    
    np.random.seed(None)

    return x, y, -z

def point_ns_fence_rng_scan(t, angles, dwell_time, state, **kw):
    '''Pointing function for a uniform radnom north-south fence scan using a fixed state at epoch, i.e reproducable sequence. NED coordinate system.
    '''
    t_n = int(np.floor(t/dwell_time))

    np.random.seed(state + t_n)
    ind = np.random.randint(len(angles))
    np.random.seed(None)

    angle = angles[int(ind)]
    no = np.cos(np.pi*angle/180.0)
    d = -np.sin(np.pi*angle/180.0)
    return no, 0.0, d

def point_cross_fence_scan(t, angles, dwell_time, state, **kw):
    n_ang = len(angles)
    ind = np.floor((t/dwell_time) % n_ang)
    angle = angles[int(ind)]

    if ind <= np.round(n_ang*0.5):
        no=np.cos(np.pi*angle/180.0)
        d=-np.sin(np.pi*angle/180.0)
        e = 0.0
    else:
        e=np.cos(np.pi*angle/180.0)
        d=-np.sin(np.pi*angle/180.0)
        no = 0.0
    return no, e, d

def point_circle_fence(t, az, el, dwell_time, **kw):
    ind = int(np.floor((t/dwell_time) % len(az)))
    return az[ind], el[ind], 1.0



def calculate_fence_angles(min_el, angle_step):
    '''Calculate a vector of angles to be used in a fence scan.
    '''
    angular_extent=180.0-2*min_el
    n_pos=int(angular_extent/angle_step)
    angles=np.arange(n_pos)*angle_step+min_el
    return angles


#below are models

def beampark_model(az, el, lat, lon, alt, name="Beampark"):
    '''A beampark model.
    
        :param float az: Azimuth in degrees east of north.
        :param float el: Elevation in degrees above horizon.
        :param float lat: Geographical latitude of radar system in decimal degrees  (North+).
        :param float lon: Geographical longitude of radar system in decimal degrees (East+).
        :param float alt: Geographical altitude above geoid surface of radar system in meter.
    '''
    beampark = rs.RadarScan(
        lat=lat,
        lon=lon,
        alt=alt,
        pointing_function=point_beampark,
        min_dwell_time=10.0,
        name=name,
        pointing_coord='azel'
    )
    beampark.keyword_arguments(
        az = az,
        el = el,
    )
    beampark._info_str = 'Dwell time: Inf, Pointing: azimuth=%.2f deg,elevation=%.2f deg' % (az, el,)
    return beampark

def n_const_pointing_model(az, el, lat, lon, alt, dwell_time = 0.1):
    '''Model for a n-point beampark with fixed dwell-times.
    
    :param list az: Azimuths in degrees east of north.
    :param list el: Elevations in degrees above horizon.
    :param float lat: Geographical latitude of radar system in decimal degrees  (North+).
    :param float lon: Geographical longitude of radar system in decimal degrees (East+).
    :param float alt: Geographical altitude above geoid surface of radar system in meter.
    :param float dwell_time: Time spent at each azimuth-elevation pair of pointing direction.
    '''
    assert len(az) == len(el), 'Length of az and el lists do not match ({} and {})'.format(len(az), len(el))
    
    n_const_pointing = rs.RadarScan(
        lat = lat,
        lon = lon,
        alt = alt,
        pointing_function = point_n_beampark,
        min_dwell_time=dwell_time,
        name = '%i beampark scan' %(len(az),),
        pointing_coord = 'azel',
    )
    n_const_pointing.keyword_arguments(
        dwell_time = dwell_time,
        az = az,
        el = el,
    )
    n_const_pointing._scan_time = len(az)*dwell_time
    return n_const_pointing

def n_dyn_dwell_pointing_model(az, el, dwells, lat, lon, alt):
    '''Model for a n-point beampark with variable dwell-times.
    
    :param list az: Azimuths in degrees east of north.
    :param list el: Elevations in degrees above horizon.
    :param list dwells: Times to spend at each azimuth-elevation pair of pointing direction.
    :param float lat: Geographical latitude of radar system in decimal degrees  (North+).
    :param float lon: Geographical longitude of radar system in decimal degrees (East+).
    :param float alt: Geographical altitude above geoid surface of radar system in meter.
    '''
    n_dyn_pointing = rs.RadarScan(
        lat = lat,
        lon = lon,
        alt = alt,
        pointing_function = point_n_dyn_beampark,
        min_dwell_time=np.min(dwells),
        name = 'Dynamic dwell scan',
        pointing_coord = 'azel',
    )
    n_dyn_pointing._scan_time = np.sum(dwells)

    n_dyn_pointing.keyword_arguments(
        dwell_time = dwells,
        az = az,
        el = el,
        scan_time = n_dyn_pointing._scan_time,
    )
    
    return n_dyn_pointing

def sph_rng_model(lat, lon, alt, min_el = 30, dwell_time = 0.1):
    sph_rng = rs.RadarScan(
        lat = lat,
        lon = lon,
        alt = alt,
        pointing_function = point_sph_rng_scan,
        min_dwell_time = dwell_time,
        name = 'Spherical random scan',
        pointing_coord = 'ned',
    )
    sph_rng.keyword_arguments(
        dwell_time = dwell_time,
        min_el = min_el/180.0*np.pi,
        state = 86246443,
    )
    sph_rng._info_str = 'Dwell time: %.2f s, Minimum elevation=%.2f deg' % (dwell_time, min_el,)
    sph_rng._scan_time = dwell_time*1000.0
    return sph_rng

def ew_fence_model(lat, lon, alt, min_el = 30, angle_step = 1.0, dwell_time = 0.1):
    ew_fence = rs.RadarScan(
        lat = lat,
        lon = lon,
        alt = alt,
        pointing_function = point_ew_fence_scan,
        min_dwell_time = dwell_time,
        name = 'East west fence scan',
        pointing_coord = 'ned',
    )
    angles = calculate_fence_angles(min_el = min_el, angle_step = angle_step)
    ew_fence.keyword_arguments(
        dwell_time = dwell_time,
        angles = angles,
    )

    ew_fence._scan_time = len(angles)*dwell_time
    ew_fence._info_str = 'Dwell time: %.2f s, Scan time: %.2f s, Minimum elevation=%.2f deg' % (dwell_time, ew_fence._scan_time, min_el,)
    return ew_fence

def ns_fence_model(lat, lon, alt, min_el = 30, angle_step = 1, dwell_time = 0.1):
    ns_fence = ew_fence_model(min_el = min_el, angle_step = angle_step, dwell_time = dwell_time, lat = lat, lon = lon, alt = alt)
    ns_fence._pointing_function = point_ns_fence_scan
    ns_fence.name = 'North south fence scan'
    return ns_fence

def ns_fence_rng_model(lat, lon, alt, min_el = 30, angle_step = 1, dwell_time = 0.1):
    ns_fence = ew_fence_model(min_el = min_el, angle_step = angle_step, dwell_time = dwell_time, lat = lat, lon = lon, alt = alt)
    ns_fence._pointing_function = point_ns_fence_rng_scan

    ns_fence.keyword_arguments(
        state = 86246443,
    )

    ns_fence.name = 'Random north south fence scan'
    return ns_fence

def flat_grid_model(lat, lon, alt, n_side = 3, height = 300e3, side_len = 100e3, dwell_time = 0.4):
    x = np.linspace(-side_len*0.5, side_len*0.5, num=n_side, dtype=np.float)
    y = np.linspace(-side_len*0.5, side_len*0.5, num=n_side, dtype=np.float)
    
    xmat, ymat = np.meshgrid(x, y, sparse=False, indexing='ij')
    
    az_points = []
    el_points = []
    for xi in range(n_side):
        for yi in range(n_side):
            az, el, r = coord.cart_to_azel([xmat[xi,yi], ymat[xi,yi], height])
            az_points.append(az)
            el_points.append(el)
            
    
    grid_scan = n_const_pointing_model(lat=lat, lon=lon, alt=alt, az = az_points, el = el_points, dwell_time = dwell_time)
    
    return grid_scan
    