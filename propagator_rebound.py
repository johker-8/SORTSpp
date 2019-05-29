#!/usr/bin/env python

'''Wrapper for the REBOUND propagator into SORTS++ format.
'''

import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from propagator_base import PropagatorBase
import dpt_tools as dpt

try:
    from propagator_orekit import frame_conversion, _get_frame
except ImportError:
    frame_conversion, _get_frame = None, None

try:
    import rebound
    import spiceypy as spice

    spice.furnsh("./data/spice/MetaK.txt")
except ImportError:
    rebound = None
    spice = None


class PropagatorRebound(PropagatorBase):
    '''Propagator class implementing the REBOUND propagator.
    
    A number of names are automatically recognized by the frame subsystem because the definitions for these frames are ``built into'' CSPICE software. Among these frames are:

    -- inertial frames such as Earth mean equator and equinox of J2000 frame ('J2000'), Mean ecliptic and equinox of J2000 ('ECLIPJ2000'), Galactic System II frame ('GALACTIC'), Mars Mean Equator and IAU vector of J2000 frame ('MARSIAU'), etc. For the complete list of ``built in'' inertial reference frames refer to the appendix ``built in Inertial Reference Frames'' of this document. 

    -- body-fixed frames based on IAU rotation models provided in text PCK files, such as Earth body-fixed rotating frame ('IAU_EARTH') and Mars body-fixed rotating frame ('IAU_MARS'), and body-fixed frames based on high precision Earth rotation models provided in binary PCK files such as 'ITRF93'. For the complete lists of ``built in'' body-fixed reference frames refer to the appendixes ``built in PCK-Based Reference Frames'' and High Precision Earth Fixed Frames'' of this document. 
    
    Heliocentric is self explaintory

    All frames that do not start with Heliocentric are considerd earth centric
    
    EME = EarthcentricJ2000
    HeliocentricECLIPJ2000 = earth orbital plane ect


    '''

    def __init__(self,
                in_frame='HeliocentricJ2000',
                out_frame='EME',
                integrator='IAS15',
                time_step=60.0,
            ):
        super(PropagatorRebound, self).__init__()
        self.planets = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
        self.planets_mass = [0.330104e24, 4.86732e24, 5.97219e24, 0.641693e24, 1898.13e24, 568.319e24, 86.8103e24, 102.410e24]
        self.m_sun = 1.98855e30
        self._earth_ind = 3
        self.N_massive = len(self.planets) + 1;
        self.integrator = integrator
        self.in_frame = in_frame
        self.out_frame = out_frame
        self.time_step = time_step

        if in_frame[:12].lower() == 'heliocentric':
            self._inframe_to_heliocentric = False
            self.in_frame = in_frame[12:]
        else:
            self._inframe_to_heliocentric = True

        if out_frame[:12].lower() == 'heliocentric':
            self._outframe_to_earthcentric = False
            self.out_frame = out_frame[12:]
        else:
            self._outframe_to_earthcentric = True

        if self.out_frame in ['ECLIPJ2000', 'J2000']:
            self.out_frame = 'J2000'
            self.orekit_out_frame = None
        else:
            self.orekit_out_frame = _get_frame(self.out_frame)

        if self.in_frame == 'ECLIPJ2000':
            self._spice_eclip_to_equat = True
            self.in_frame = 'EME'
        else:
            self._spice_eclip_to_equat = False

        if self.in_frame == 'J2000':
            self.in_frame = 'EME'
        self.orekit_in_frame = _get_frame(self.in_frame)
        

        if self.in_frame == 'EME':
            self.inertialFrame = self.orekit_in_frame
        else:
            self.inertialFrame = _get_frame('EME')


    def _setup_sim(self, mjd0):
        self.sim = rebound.Simulation()
        self.sim.units = ('m', 's', 'kg')
        self.sim.integrator = self.integrator
        self.et = dpt.mjd_to_j2000(mjd0)*3600.0*24.0
        
        self.sim.add(m=self.m_sun)
        for i in range(0,len(self.planets)):
            #Units are always km and km/sec. 
            state, lightTime = spice.spkezr(
                self.planets[i] + ' BARYCENTER',
                self.et,
                'J2000',
                'NONE',
                'SUN',
            )
            self.sim.add(m=self.planets_mass[i],
                x=state[0]*1e3,  y=state[1]*1e3,  z=state[2]*1e3,
                vx=state[3]*1e3, vy=state[4]*1e3, vz=state[5]*1e3,
            )
        self.sim.N_active = self.N_massive
        self.sim.dt = self.time_step


    def _add_state(self, states, ti):
        particle = self.sim.particles[self.N_massive]
        states[0,ti] = particle.x
        states[1,ti] = particle.y
        states[2,ti] = particle.z
        states[3,ti] = particle.vx
        states[4,ti] = particle.vy
        states[5,ti] = particle.vz

    def get_orbit(self,t,a,e,inc,raan,aop,mu0,mjd0, **kwargs):
        '''
        **Implementation:**
    
        Units are in meters and degrees.

        Keyword arguments are:

        **Uses:**
            * 
        
        See :func:`propagator_base.PropagatorBase.get_orbit`.
        '''

        m = kwargs.setdefault('m', 0.0)

        orb = np.array([a,e,inc,aop,raan,dpt.mean2true(mu0,e, radians = False)], dtype=np.float)
        
        if self._inframe_to_heliocentric:
            M_cent = self.planets_mass[self._earth_ind - 1]
        else:
            M_cent = self.m_sun

        x, y, z, vx, vy, vz = dpt.kep2cart(
            orb,
            m=m,
            M_cent=M_cent,
            radians=False,
        )

        return self.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0, **kwargs)

    def _get_earth_state(self):
        earth_state = np.empty((6,), dtype=np.float64)
        earth = self.sim.particles[self._earth_ind]
        earth_state[0] = earth.x
        earth_state[1] = earth.y
        earth_state[2] = earth.z
        earth_state[3] = earth.vx
        earth_state[4] = earth.vy
        earth_state[5] = earth.vz
        return earth_state

    def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
        '''
        **Implementation:**
    
        Units are in meters and degrees.

        Keyword arguments are:

        **Uses:**
            * 
        
        See :func:`propagator_base.PropagatorBase.get_orbit`.
        '''
        t = self._make_numpy(t)

        state = np.array([x,y,z,vx,vy,vz], dtype=np.float)

        t_order = np.argsort(t)
        t_restore = np.argsort(t_order)

        t = t[t_order]

        self._setup_sim(mjd0)

        m = kwargs.setdefault('m', 0.0)

        N_testparticle = 1

        if self._spice_eclip_to_equat and not self._inframe_to_heliocentric:
            M_trf = spice.pxform('ECLIPJ2000', 'J2000', self.et)
            #print('from ECLIPJ2000 to J2000')
            state[:3] = M_trf.dot(state[:3])
            state[3:] = M_trf.dot(state[3:])

        if self.in_frame not in ['EME', 'J2000'] and self._inframe_to_heliocentric:
            #print('from {} to {}'.format(self.in_frame, 'EME'))
            state = frame_conversion(state, mjd0, self.orekit_in_frame, self.inertialFrame)

        if self._inframe_to_heliocentric:
            #print('from {} to {}'.format('Earthcentric', 'Heliocentric'))
            state += self._get_earth_state()

        x, y, z, vx, vy, vz = state

        self.sim.add(
            x = x,
            y = y,
            z = z,
            vx = vx,
            vy = vy,
            vz = vz,
            m = m,
        )

        states = np.empty((6, len(t)), dtype=np.float64)


        #self.sim.move_to_com()
        for ti in range(len(t)):

            self.sim.integrate(t[ti])
            self._add_state(states, ti)

            if self._outframe_to_earthcentric:
                states[:,ti] -= self._get_earth_state()

        #if self._outframe_to_earthcentric:
        #    print('from {} to {}'.format('Heliocentric', 'Earthcentric'))

        if self.out_frame not in ['EME', 'J2000'] and self._outframe_to_earthcentric:
            #print('from {} to {}'.format('EME', self.out_frame))
            states = frame_conversion(states, t/(3600.0*24.0) + mjd0, self.inertialFrame, self.orekit_out_frame)

        states = states[:, t_restore]

        return states



