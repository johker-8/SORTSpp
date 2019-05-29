#!/usr/bin/env python

'''Kepler propagation interface with SORTS++.
'''

import numpy as np
import scipy.constants as consts

import dpt_tools as dpt
from propagator_base import PropagatorBase
from propagator_sgp4 import M_earth

try:
    from propagator_orekit import frame_conversion, _get_frame
except ImportError:
    frame_conversion = None


class PropagatorKepler(PropagatorBase):
    '''Propagator class implementing a analytic Kepler propagator.

    The constructor creates a propagator instance.

    :ivar str in_frame: String identifying the input frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :ivar str out_frame: String identifying the output frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :ivar bool frame_tidal_effects: Should coordinate frames include Tidal effects.

    :param str in_frame: String identifying the input frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :param str out_frame: String identifying the output frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :param bool frame_tidal_effects: Should coordinate frames include Tidal effects.
    '''

    def __init__(self, in_frame='EME', out_frame='ITRF', frame_tidal_effects=False):
        super(PropagatorKepler, self).__init__()
        self.in_frame = in_frame
        self.out_frame = out_frame
        self.frame_tidal_effects = frame_tidal_effects

        self.orekit_in_frame = _get_frame(in_frame)
        self.orekit_out_frame = _get_frame(out_frame)


    def output_to_input_frame(self, states, t, mjd0):
        return frame_conversion(states, t/(3600.0*24.0) + mjd0, self.orekit_out_frame, self.orekit_in_frame)

    def get_orbit(self, t, a, e, inc, raan, aop, mu0, mjd0, **kwargs):
        '''
        **Implementation:**
    
        All state-vector units are in meters.

        Keyword arguments contain only mass :code:`m` in kg and is not required.

        They also contain a option to give angles in radians or degrees. By default input is assumed to be degrees.

        **Frame:**

        The input frame is always the same as the output frame.
    
        :param float m: Mass of the object in kg.
        :param bool radians: If true, all angles are assumed to be in radians.
        '''

        radians = kwargs.setdefault('radians', False)
        m = kwargs.setdefault('m', 0.0)

        if radians:
            one_lap = np.pi*2.0
        else:
            one_lap = 360.0

        gravitational_param = consts.G*(M_earth + m)

        mean_motion = np.sqrt(gravitational_param/(a**3.0))/(np.pi*2.0)

        t = self._make_numpy(t)

        mean_anomalies = mu0 + t*mean_motion*one_lap
        mean_anomalies = np.remainder(mean_anomalies, one_lap)

        true_anomalies = dpt.mean2true(mean_anomalies, e, radians=radians)

        orb0 = np.array([a, e, inc, aop, raan], dtype=np.float64)
        orb0.shape = (5,1)

        orbs = np.empty((6, len(t)), dtype=np.float64)
        orbs[:5, :] = np.repeat(orb0, len(t), axis=1)
        orbs[5, :] = true_anomalies

        states_raw = dpt.kep2cart(orbs, m=m, M_cent=M_earth, radians=radians)

        if self.in_frame != self.out_frame:
            states = frame_conversion(states_raw, t/(3600.0*24.0) + mjd0, self.orekit_in_frame, self.orekit_out_frame)
        else:
            states = states_raw
        
        return states




    def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
        '''
        **Implementation:**
    
        All state-vector units are in meters.

        Keyword arguments contain only mass :code:`m` in kg and is not required.

        They also contain a option to give angles in radians or degrees. By default input is assumed to be degrees.

        **Frame:**

        The input frame is always the same as the output frame.
    
        :param float m: Mass of the object in kg.
        :param bool radians: If true, all angles are assumed to be in radians.
        '''

        radians = kwargs.setdefault('radians', False)
        m = kwargs.setdefault('m', 0.0)

        cart0 = np.array([x, y, z, vx, vy, vz], dtype=np.float64)
        orb0 = dpt.cart2kep(cart0, m=m, M_cent=M_earth, radians=radians)
        a = orb0[0]
        e = orb0[1]
        inc = orb0[2]
        aop = orb0[3]
        raan = orb0[4]
        mu0 = dpt.true2mean(orb0[5], orb0[1], radians=radians)

        return self.get_orbit(t, a, e, inc, raan, aop, mu0, mjd0, **kwargs)

