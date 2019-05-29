#!/usr/bin/env python

'''Defines what a radar observation schema is in the form of a class.

A scan needs to return:
 * radar position and radar pointing direction at any given time t (seconds since epoch)
 * short name of scan
 * title describing the scan



'''

import numpy as n
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import copy

import coord
import plothelp


class RadarScan(object):
    '''Encapsulates the observation schema of a radar system, i.e. its "scan".
    
    :ivar float _lat: Geographical latitude of radar system in decimal degrees  (North+).
    :ivar float _lon: Geographical longitude of radar system in decimal degrees (East+).
    :ivar float _alt: Geographical altitude above geoid surface of radar system in meter.
    :ivar function _pointing_function: A function that takes in time as the first argument and then any number of keyword arguments and returns the pointing of the radar in a system specified by :code:`pointing_coord`.
    :ivar str _pointing_coord: The coordinate system used by the :code:`_pointing_function`, may be 'azel' or 'ned'.
    :ivar str name: Name of the scan.
    :ivar dict _function_data: A dictionary contaning the data to be expanded as keyword parameters to the :code:`_pointing_function`.
    :ivar str _info_str: A string describing the scan.
    :ivar float _scan_time: If the scan has a repeating deterministic sequence, it is he time it takes to complete one sequence.
    :ivar float _pulse_n: Number of pulses in a repeating pulse sequence.
    :ivar float _min_el: Minimum elevation of the scanning sequence.
    
    
    :param float lat: Geographical latitude of radar system in decimal degrees  (North+).
    :param float lon: Geographical longitude of radar system in decimal degrees (East+).
    :param float alt: Geographical altitude above geoid surface of radar system in meter.
    :param float pointing_function: A function that takes in time as the first argument and then any number of keyword arguments and returns the pointing of the radar in a system specified by :code:`pointing_coord`.
    :param str pointing_coord: The coordinate system used by the :code:`_pointing_function`, may be 'azel' or 'ned'.
    :param str name: Name of the scan.
    
    **Pointing function:**
    
    The pointing function must follow the following standard:

     * Take in time in seconds past reference epoch in seconds as first argument
     * Take any number of keyword arguments, these arguments must be defined in the :code:`_function_data` dictionary.
     * It must return the pointing coordinates as a object with get-item implemented (list, tuple, 1-D numpy array, ect) of 3 elements.
     * Units are in meters or degrees.
     
    Example pointing function:
    
    .. code-block:: python

        import numpy as np
        
        def point_east_west_fence(t, dwell_time, angles):
            """Pointing function for a east-to-west fence scan returning pointing coordinates in a NED (North-East-Down) cartesian coordinate system.
            """
        	ind = np.floor(t/dwell_time % len(angles))
        	angle = int(ind)
        	e = np.cos(np.pi*angle/180.0)
        	d = -np.sin(np.pi*angle/180.0)
        	return 0.0, e, d


    **Coordinate systems:**

     :azel: Azimuth and Elevation in degrees east of north and above horizon.
     :ned: Cartesian coordinates in North, East, Down in meters.
     :enu: Cartesian coordinates in East, North, Up in meters.


    '''
    def __init__(self, lat, lon, alt, pointing_function, min_dwell_time, pointing_coord='azel', name='generic scan'):
        self._lat=lat
        self._lon=lon
        self._alt=alt
        self._pointing_function = pointing_function
        self._pointing_coord = pointing_coord
        self.name = name
        self._function_data = {}

        self.min_dwell_time = min_dwell_time

        self._info_str = ''
        
        self._scan_time = None
        self._pulse_n = None
        self._min_el = None

    @property
    def min_dwell_time(self):
        '''The dwell time of the scan. If there are dynamic dwell times, this is the minimum dwell time.
        '''
        return self._function_data['min_dwell_time']


    @min_dwell_time.setter
    def min_dwell_time(self, dw):
        self._function_data['min_dwell_time'] = dw


    @min_dwell_time.deleter
    def min_dwell_time(self):
        del self._function_data['min_dwell_time']


    def info(self):
        '''Return a descriptive string.
        '''
        return self._info_str

    def keyword_arguments(self, **kw):
        '''Adds or modifies all the input keyword arguments of this call to the function data used in calling the pointing function.
        '''
        for key, val in kw.items():
            self._function_data[key] = val

    def dwell_time(self):
        '''If dwell time is a applicable concept for this scan, return that time.
        '''
        if 'dwell_time' in self._function_data:
            return self._function_data['dwell_time']
        else:
            return None

    def copy(self):
        '''Return a copy of the current instance of :class:`radar_scans.RadarScan`.
        '''
        C = RadarScan(
            lat = self._lat,
            lon = self._lon,
            alt = self._alt,
            pointing_function = self._pointing_function,
            min_dwell_time = self.min_dwell_time,
            pointing_coord = self._pointing_coord,
            name = self.name,
        )
        
        C._function_data = copy.deepcopy(self._function_data)

        C._scan_time = self._scan_time
        C._pulse_n = self._pulse_n
        C._min_el = self._min_el
        C._info_str = self._info_str

        return C

    def set_tx_location(self, tx):
        '''Set the geographic location of this scan to coencide with the input :class:`antenna.AntennaTX`.
        
            :param AntennaTX tx: The antenna that should perform this scan.
        '''
        self._lat = tx.lat
        self._lon = tx.lon
        self._alt = tx.alt
    
    def check_tx_compatibility(self, tx):
        '''Checks if the transmitting antenna pusle pattern and coherrent integration schema is compatible with the observation schema. Raises an Exception if not.
        
            :param AntennaTX tx: The antenna that should perform this scan.
        '''
        time_slice = tx.n_ipp*tx.ipp
        dwell_time = self.dwell_time()
        if dwell_time is not None:
            try:
                iter(dwell_time)
                is_iter = True
            except TypeError:
                is_iter = False
            
            if is_iter:
                for dw in dwell_time:
                    RadarScan._check_dw(time_slice, dw)
            else:
                RadarScan._check_dw(time_slice, dwell_time)
                
    @staticmethod
    def _check_dw(time_slice, dwell_time):
        if time_slice > dwell_time:
                raise Exception('TX "{}" cannot complete coherrent integration during scan "{}": dwell time = {} s, time slice = {} s'.format(tx.name, self.name, dwell_time, time_slice))


    def _pointing(self,t):
        return self._pointing_function(t, **self._function_data)


    def local_pointing(self,t):
        '''Returns the instantaneous pointing in local coordinates (ENU).
        
            :param float t: Seconds past a reference epoch to retrieve the pointing at.
        '''
        point = self._pointing(t)

        if self._pointing_coord == 'ned':
            k0 = n.array([point[1], point[0], -point[2]], dtype=n.float)
        elif self._pointing_coord == 'enu':
            k0 = n.array([point[0], point[1], point[2]], dtype=n.float)
        elif self._pointing_coord == 'azel':
            k0 = coord.azel_to_cart(point[0], point[1], 1.0)

        return k0
    

    def antenna_pointing(self,t):
        '''Returns the instantaneous WGS84 ECEF pointing direction and the radar geographical location in WGS84 ECEF coordinates.
        
            :param float t: Seconds past a reference epoch to retrieve the pointing at.
        '''
        p0 = coord.geodetic2ecef(self._lat, self._lon, self._alt)

        point = self._pointing(t)

        if self._pointing_coord == 'ned':
            k0 = coord.ned2ecef(self._lat, self._lon, self._alt, point[0],point[1],point[2])
        elif self._pointing_coord == 'enu':
            k0 = coord.enu2ecef(self._lat, self._lon, self._alt, point[0],point[1],point[2])
        elif self._pointing_coord == 'azel':
            k0 = coord.azel_ecef(self._lat, self._lon, self._alt, point[0], point[1])
        
        return p0, k0

def plot_radar_scan(SC, earth=False, ax = None):
    '''Plot a full cycle of the scan pattern based on the :code:`_scan_time` and the :code:`_function_data['dwell_time']` variable.
    
        :param RadarScan SC: Scan to plot.
        :param bool earth: Plot the surface of the Earth.
    '''
    if 'dwell_time' in SC._function_data:
        dwell_time = n.min(SC._function_data['dwell_time'])
    else:
        dwell_time = 0.05
    
    if SC._scan_time is None:
        scan_time = dwell_time*100.0
    else:
        scan_time = SC._scan_time
    
    t=n.linspace(0.0, scan_time, num=n.round(2*scan_time/dwell_time))

    if ax is None:
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)
        ax.view_init(15, 5)
            
        plt.title(SC.name)
        plt.tight_layout()
        _figs = (fig, ax)
    else:
        _figs = (None, ax)

    if earth:
        plothelp.draw_earth_grid(ax)
    plothelp.draw_radar(ax,SC._lat,SC._lon)
    
    max_range=4000e3
    
    for i in range(len(t)):
        p0,k0=SC.antenna_pointing(t[i])
        
        p1=p0+k0*max_range*0.8
        if k0[2] < 0:
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="red")
        else:
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="green")
    
    ax.set_xlim(p0[0] - max_range, p0[0] + max_range)
    ax.set_ylim(p0[1] - max_range, p0[1] + max_range)
    ax.set_zlim(p0[2] - max_range, p0[2] + max_range)


    return _figs

def plot_radar_scan_movie(SC, earth=False, rotate=False, save_str=''):
    '''Create a animation of the scan pattern based on the :code:`_scan_time` and the :code:`_function_data['dwell_time']` variable.
    
        :param RadarScan SC: Scan to plot.
        :param bool earth: Plot the surface of the Earth.
        :param str save_str: String of path to output movie file. Requers an avalible ffmpeg encoder on the system. If string is empty no movie is saved.
    '''
    if 'dwell_time' in SC._function_data:
        dwell_time = n.min(SC._function_data['dwell_time'])
    else:
        dwell_time = 0.05
    
    if SC._scan_time is None:
        scan_time = dwell_time*100.0
    else:
        scan_time = SC._scan_time
    
    t=n.linspace(0.0, scan_time, num=n.round(2*scan_time/dwell_time))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)

    def update_text(SC,t):
        return SC.name + ', t=%.4f s' % (t*1e0,)

    titl = fig.text(0.5,0.94,update_text(SC,t[0]),size=22,horizontalalignment='center')


    max_range=4000e3

    p0,k0=SC.antenna_pointing(0)
    p1=p0+k0*max_range*0.8

    if earth:
        plothelp.draw_earth_grid(ax)
    else:
        plothelp.draw_earth(ax)
    plothelp.draw_radar(ax,SC._lat,SC._lon)
    if k0[2] < 0:
        beam = ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="red")
    else:
        beam = ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],alpha=0.5,color="green")

    ax.set_xlim(p0[0] - max_range, p0[0] + max_range)
    ax.set_ylim(p0[1] - max_range, p0[1] + max_range)
    ax.set_zlim(p0[2] - max_range, p0[2] + max_range)

    interval = scan_time*1e3/float(len(t))
    rotations = n.linspace(0.,360.*2, num=len(t)) % 360.0
    
    def update(ti,beam):
        _t = t[ti]
        p0,k0=SC.antenna_pointing(_t)
        p1=p0+k0*max_range*0.8
        titl.set_text(update_text(SC,_t))
        beam.set_data([p0[0],p1[0]],[p0[1],p1[1]])
        beam.set_3d_properties([p0[2],p1[2]])
        if k0[2] < 0:
            beam.set_color("red")
        else:
            beam.set_color("green")
        
        if rotate:
            ax.view_init(15, rotations[ti])
        
        return beam,
    
    ani = animation.FuncAnimation(fig, update, frames=range(len(t)), fargs=(beam), interval=interval, blit=False)

    if len(save_str)>0:

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(metadata=dict(artist='Daniel Kastinen'), bitrate=1800)
        ani.save(save_str, writer=writer)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pass