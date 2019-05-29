#!/usr/bin/env python

'''A parent class used for interfacing any propagator.

'''

from abc import ABCMeta, abstractmethod
import inspect

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import plothelp

class PropagatorBase:
    __metaclass__ = ABCMeta

    def __init__(self):
        self._check_args()


    def _check_args(self):
        '''This method makes sure that the function signature of the implemented :func:`propagator_base.PropagatorBase.get_orbit_cart` and :func:`propagator_base.PropagatorBase.get_orbit_cart` are correct.
        '''
        correct_argspec = inspect.getargspec(PropagatorBase.get_orbit)
        current_argspec = inspect.getargspec(self.get_orbit)

        correct_vars = correct_argspec.args
        current_vars = current_argspec.args

        assert len(correct_vars) == len(current_vars), 'Number of arguments in implemented get_orbit is wrong ({} not {})'.format(current_vars, correct_vars)
        for var in current_vars:
            assert var in correct_vars, 'Argument missing in implemented get_orbit, got "{}" instead'.format(var)


    def _make_numpy(self, var):
        '''Small method for converting non-numpy data structures to numpy data arrays. 
        Should be used at top of functions to minimize type checks and maximize computation speed by avoiding Python objects.
        '''
        if not isinstance(var, np.ndarray):
            if isinstance(var, float):
                var = np.array([var], dtype=np.float)
            elif isinstance(var, list):
                var = np.array(var, dtype=np.float)
            else:
                raise Exception('Input type {} not supported'.format(type(var)))
        return var


    @abstractmethod
    def get_orbit(self, t, a, e, inc, raan, aop, mu0, mjd0, **kwargs):
        '''Propagate a Keplerian state forward in time.

        This function uses key-word argument to supply additional information to the propagator, such as area or mass.

        It is a good idea to only implement :func:`propagator_base.PropagatorBase.get_orbit` or :func:`propagator_base.PropagatorBase.get_orbit_cart` and then link one to the other by simply using :func:`dpt_tools.kep2cart` or :func:`dpt_tools.cart2kep`.
        
        The coordinate frames used should be documented in the child class docstring.

        SI units are assumed unless implementation states otherwise.

        :param float/list/numpy.ndarray t: Time in seconds to propagate relative the initial state epoch.
        :param float mjd0: The epoch of the initial state in fractional Julian Days.
        :param float a: Semi-major axis
        :param float e: Eccentricity
        :param float inc: Inclination
        :param float aop: Argument of perihelion
        :param float raan: Longitude (right ascension) of ascending node
        :param float mu0: Mean anomaly
        :return: 6-D Cartesian state vector in SI-units.
        '''
        return None

    @abstractmethod
    def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
        '''Propagate a Cartesian state forward in time.

        This function uses key-word argument to supply additional information to the propagator, such as area or mass.

        It is a good idea to only implement :func:`propagator_base.PropagatorBase.get_orbit` or :func:`propagator_base.PropagatorBase.get_orbit_cart` and then link one to the other by simply using :func:`dpt_tools.kep2cart` or :func:`dpt_tools.cart2kep`.
        
        The coordinate frames used should be documented in the child class docstring.

        :param float/list/numpy.ndarray t: Time in seconds to propagate relative the initial state epoch.
        :param float mjd0: The epoch of the initial state in fractional Julian Days.
        :param float x: X position
        :param float y: Y position
        :param float z: Z position
        :param float vx: X-direction velocity
        :param float vy: Y-direction velocity
        :param float vz: Z-direction velocity
        :return: 6-D Cartesian state vector in SI-units.
        '''
        return None


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        pass 


def plot_orbit_3d(ecefs):
    '''Plot a set of ECEF's in 3D using matplotlib.
    '''
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],"-",alpha=0.75,color="black")
    plt.title("Orbital propagation")
    plt.show()
