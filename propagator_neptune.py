#!/usr/bin/env python

'''Neptune interface with SORTS++.

sudo apt-get install libopenmpi-dev libmpich-dev
'''

import neptune as _neptune

import numpy as np
import scipy.constants as consts

import dpt_tools as dpt
from propagator_base import PropagatorBase
from propagator_sgp4 import M_earth

class PropagatorNeptune(PropagatorBase):
    '''Neptune propagator
    '''

    def __init__(self):
        super(PropagatorNeptune, self).__init__()

    def get_orbit(self, t, a, e, inc, raan, aop, mu0, mjd0, **kwargs):
        '''
        '''
        return _neptune.m_get_orbit(
            t = t,
            a0 = a*1e-3,
            e0 = e,
            inc0 = inc,
            raan0 = raan,
            aop0 = aop,
            mu00 = mu0,
            mjd0 = mjd0,
            C_D = kwargs.setdefault('C_D', 2.3),
            A = kwargs['A'],
            m = kwargs['m'],
            host = _neptune.host,
            OPI_propagator = _neptune.OPI_propagator,
        )


    def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
        '''
        **Implementation:**

        Converts Cartesian vector to Kepler elements and calls :func:`propagator_neptune.PropagatorNeptune.get_orbit`.

        All units are in m and m/s.

        **Uses:**
            * :func:`dpt_tools.cart2kep`
            * :func:`dpt_tools.true2mean`

        See :func:`propagator_base.PropagatorBase.get_orbit_cart`.
        '''

        cart = np.array([x,y,z,vx,vy,vz], dtype=np.float)
        orb = dpt.cart2kep(
            cart,
            m=kwargs['m'],
            M_cent=M_earth,
            radians=False,
        )

        a = orb[0]
        e = orb[1]
        inc = orb[2]
        aop = orb[3]
        raan = orb[4]
        mu0 = dpt.true2mean(orb[5], e, radians=False)

        return self.get_orbit(t, a, e, inc, raan, aop, mu0, mjd0, **kwargs)

