#!/usr/bin/env python

'''Defines a space object. Encapsulates orbital elements, propagation and related methods.


**Example:**

Using space object for propagation.

.. code-block:: python

    import numpy as n
    import matplotlib.pyplot as plt
    import SpaceObject as so
    import plothelp

    o = so.SpaceObject(
        a=7000, e=0.0, i=69,
        raan=0, aop=0, mu0=0,
        C_D=2.3, A=1.0, m=1.0,
        C_R=1.0, oid=42,
        mjd0=57125.7729,
    )

    t=n.linspace(0,24*3600,num=1000, dtype=n.float)
    ecefs=o.get_state(t)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)
    plothelp.draw_earth_grid(ax)

    ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],'-',alpha=0.5,color="black")
    plt.title("Orbital propagation test")
    plt.show()


Using space object with a different propagator.

.. code-block:: python

    import numpy as n
    import matplotlib.pyplot as plt
    import SpaceObject as so
    import plothelp
    from propagator_orekit import PropagatorOrekit

    o = so.SpaceObject(
        a=7000, e=0.0, i=69,
        raan=0, aop=0, mu0=0,
        C_D=2.3, A=1.0, m=1.0,
        C_R=1.0, oid=42,
        mjd0=57125.7729,
        propagator = PropagatorOrekit,
        propagator_options = {
            'in_frame': 'TEME',
            'out_frame': 'ITRF',
        },
    )

    t=n.linspace(0,24*3600,num=1000, dtype=n.float)
    ecefs=o.get_state(t)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)
    plothelp.draw_earth_grid(ax)

    ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],'-',alpha=0.5,color="black")
    plt.title("Orbital propagation test")
    plt.show()


'''
import os

import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sorts_config import p as default_propagator
import plothelp
import ccsds_write
import dpt_tools as dpt
import TLE_tools as tle

M_e = 5.972e24
'''float: Mass of the Earth
'''

R_E = 6371e3
'''float: Radius of the Earth
'''




class SpaceObject(object):
    '''Encapsulates a object in space who's dynamics is governed in time by a propagator.

    The relation between the Cartesian and Kepler states are a direct transformation according to the below orientation rules.
    If the Kepler elements are given in a Inertial system, to reference the Cartesian to a Earth-fixed system a earth rotation transformation
    must be applied externally of the method.


    **Orientation of the ellipse in the coordinate system:**
       * For zero inclination :math:`i`: the ellipse is located in the x-y plane.
       * The direction of motion as True anoamly :math:`\\nu`: increases for a zero inclination :math:`i`: orbit is anti-coockwise, i.e. from +x towards +y.
       * If the eccentricity :math:`e`: is increased, the periapsis will lie in +x direction.
       * If the inclination :math:`i`: is increased, the ellipse will rotate around the x-axis, so that +y is rotated toward +z.
       * An increase in Longitude of ascending node :math:`\Omega`: corresponds to a rotation around the z-axis so that +x is rotated toward +y.
       * Changing argument of perihelion :math:`\omega`: will not change the plane of the orbit, it will rotate the orbit in the plane.
       * The periapsis is shifted in the direction of motion.
       * True anomaly measures from the +x axis, i.e :math:`\\nu = 0` is located at periapsis and :math:`\\nu = \pi` at apoapsis.
       * All anomalies and orientation angles reach between 0 and :math:`2\pi`

       *Reference:* "Orbital Motion" by A.E. Roy.
    

    **Variables:**
       * :math:`a`: Semi-major axis
       * :math:`e`: Eccentricity
       * :math:`i`: Inclination
       * :math:`\omega`: Argument of perihelion
       * :math:`\Omega`: Longitude of ascending node
       * :math:`\\nu`: True anoamly


    **Uses:**
       * :func:`~dpt_tools.kep2cart`
       * :func:`~dpt_tools.cart2kep`
       * :func:`~dpt_tools.mean2true`
       * :func:`~dpt_tools.true2mean`
       * :func:`~dpt_tools.rot_mat_z`
       * :func:`~dpt_tools.gmst`
       * :func:`~ccsds_write.write_oem`


    :ivar float a: Semi-major axis [km]
    :ivar float e: Eccentricity
    :ivar float i: Inclination [deg]
    :ivar float aop: Argument of periapsis [deg]
    :ivar float raan: Right ascension of the ascending node [deg]
    :ivar float mu0: Mean anomaly [deg]
    :ivar float x: X position [km]
    :ivar float y: Y position [km]
    :ivar float z: Z position [km]
    :ivar float vx: X-direction velocity [km/s]
    :ivar float vy: Y-direction velocity [km/s]
    :ivar float vz: Z-direction velocity [km/s]
    :ivar int oid: Identifying object ID
    :ivar float C_D: Drag coefficient
    :ivar float C_R: Radiation pressure coefficient
    :ivar float A: Area [:math:`m^2`]
    :ivar float m: Mass [kg]
    :ivar float mjd0: Epoch for state [BC-relative JD]
    :ivar float prop: Propagator instance, child of :class:`~base_propagator.PropagatorBase`
    :ivar float d: Diameter [m]
    :ivar float M_cent: Mass of central body [kg]
    :ivar numpy.ndarray state_cart: 6-D vector containing the Cartesian state vector.
    :ivar dict propagator_options: Propagator initialization keyword arguments
    :ivar dict kwargs: All additional keyword arguments will be passed to the propagator call.

    The constructor creates a space object using Kepler elements.

    :param float a: Semi-major axis in km
    :param float e: Eccentricity
    :param float i: Inclination in degrees
    :param float aop: Argument of perigee in degrees
    :param float raan: Right ascension of the ascending node in degrees
    :param float mu0: Mean anomaly in degrees
    :param float C_D: Drag coefficient
    :param float C_R: Radiation pressure coefficient
    :param float A: Area in square meters
    :param float m: Mass in kg
    :param float mjd0: Epoch for state
    :param int oid: Identifying object ID
    :param float d: Diameter in meters
    :param float M_cent: Mass of central body
    :param PropagatorBase propagator: Propagator class pointer
    :param dict propagator_options: Propagator initialization keyword arguments
    :param dict kwargs: All additional keyword arguments will be passed to the propagator call.
    
    '''
    def __init__(self,
            a, e, i, raan, aop, mu0,
            d=0.01,
            C_D=2.3,
            A=1.0,
            m=1.0,
            mjd0=57125.7729,
            oid=42,
            M_cent = M_e,
            C_R = 1.0,
            propagator = default_propagator,
            propagator_options = {},
            **kwargs
        ):
        
        self.kwargs = kwargs

        self.M_cent = M_cent

        orb = n.array([a*1e3, e, i, aop, raan, dpt.mean2true(mu0, e, radians=False)], dtype=n.float)
        state_c = dpt.kep2cart(orb, m=m, M_cent=self.M_cent, radians=False)*1e-3

        x = state_c[0]
        y = state_c[1]
        z = state_c[2]
        vx = state_c[3]
        vy = state_c[4]
        vz = state_c[5]

        self.state_cart = state_c*1e3

        self.a=a
        self.e=e
        self.i=i
        self.oid=oid
        self.raan=raan
        self.aop=aop
        self.mu0=mu0

        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz

        self.C_D=C_D
        self.C_R=C_R
        self.A=A
        self.m=m
        self.mjd0=mjd0
        self._propagator = propagator
        self.propagator_options = propagator_options
        self.prop=propagator(**propagator_options)
        self.d=d
    
    def copy(self):
        '''Returns a copy of the SpaceObject instance.
        '''
        return SpaceObject(
            a = self.a,
            e = self.e,
            i = self.i,
            raan = self.raan,
            aop = self.aop,
            mu0 = self.mu0,
            d = self.d,
            C_D = self.C_D,
            A = self.A,
            m = self.m,
            mjd0 = self.mjd0,
            oid = self.oid,
            M_cent = self.M_cent,
            C_R = self.C_R,
            propagator = self._propagator,
            propagator_options = self.propagator_options,
            **self.kwargs
        )


    @property
    def diam(self):
        return self.d
    
    @diam.setter
    def diam(self, val):
        self.d = val

    # todo    
    # @classmethod
    #def from_oem(so_class,fname, propagator, propagator_options={}):
    #    # read oem file
    #    # figure out epoch, based on first oem point time
    #    # 
    #    # do a least-squares fit
    #    # create space object with fitted parameters
    #    # o = cls.cartesian

        
    @classmethod
    def cartesian(so_class,
            x, y, z, vx, vy, vz,
            d=0.01,
            C_D=2.3,
            A=1.0,
            m=1.0,
            mjd0=57125.7729,
            oid=42,
            M_cent = M_e,
            C_R = 1.0,
            propagator = default_propagator,
            propagator_options = {},
            **kwargs
        ):
        '''Creates a space object using Cartesian elements.

            :param float x: X position in km
            :param float y: Y position in km
            :param float z: Z position in km
            :param float vx: X-direction velocity in km/s
            :param float vy: Y-direction velocity in km/s
            :param float vz: Z-direction velocity in km/s
            :param float C_D: Drag coefficient
            :param float C_R: Radiation pressure coefficient
            :param float A: Area
            :param float m: Mass in kg
            :param float mjd0: Epoch for state
            :param int oid: Identifying object ID
            :param float d: Diameter in meters
            :param float M_cent: Mass of central body
            :param PropagatorBase propagator: Propagator class pointer
            :param dict propagator_options: Propagator initialization keyword arguments
            :param dict kwargs: All additional keyword arguments will be passed to the propagator call.
        '''

        state = n.array([x, y, z, vx, vy, vz], dtype=n.float)*1e3
        eart_rot_inv = dpt.rot_mat_z(-dpt.gmst(mjd0))
        #state[3:] = eart_rot_inv.dot(state[3:])
        #state[:3] = eart_rot_inv.dot(state[:3])
        orb_c = dpt.cart2kep(state, m=m, M_cent=M_cent, radians=False)

        a = orb_c[0]*1e-3
        e = orb_c[1]
        i = orb_c[2]
        raan = orb_c[4]
        aop = orb_c[3]
        mu0 = dpt.true2mean(orb_c[5], orb_c[1], radians=False)

        return so_class(
                    a=a, e=e, i=i,
                    raan=raan, aop=aop, mu0=mu0,
                    d=d,
                    C_D=C_D,
                    A=A,
                    m=m,
                    mjd0=mjd0,
                    oid=oid,
                    M_cent = M_cent,
                    C_R = C_R,
                    propagator = propagator,
                    propagator_options = propagator_options,
                    **kwargs
            )


    def propagate(self, dt, frame_transformation, frame_options = {}):
        '''Propagate and change the epoch of this space object

        Frame transformations available:
            * TEME: From ITRF to TEME
            * ITRF: From TEME to ITRF
            * None: Do not perform transformation
            * function pointer: use custom transformation function that takes state as first argument (in SI units) and keyword arguments
        
        '''

        state = self.get_state(np.array([dt], dtype=n.float64))
        self.mjd0 = self.mjd0 + dt/(3600.0*24.0)

        if isinstance(frame_transformation, str):
            jd_ut1 = dpt.mjd_to_jd(self.mjd0)

            frame_options.setdefault('xp', 0.0)
            frame_options.setdefault('yp', 0.0)

            if frame_transformation == 'ITRF':
                state_frame = tle.TEME_to_ITRF(state[:,0], jd_ut1, **frame_options)
            elif frame_transformation == 'TEME':
                state_frame = tle.ITRF_to_TEME(state[:,0], jd_ut1, **frame_options)
            else:
                raise ValueError('Tranformation {} not recognized'.format(frame_transformation))
        elif isinstance(frame_transformation, None):
            pass
        else:
            state_frame = frame_transformation(state, jd_ut1, **frame_options)

        x, y, z, vx, vy, vz = (state_frame*1e-3).flatten()

        self.update(
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
        )



    def update(self, **kwargs):
        '''Updates the orbital elements and Cartesian state vector of the space object.

        Can update any of the related state parameters, all others will automatically update.

        Cannot update Keplerian and Cartesian elements simultaneously.

        :param float a: Semi-major axis in km
        :param float e: Eccentricity
        :param float i: Inclination in degrees
        :param float aop: Argument of perigee in degrees
        :param float raan: Right ascension of the ascending node in degrees
        :param float mu0: Mean anomaly in degrees
        :param float x: X position in km
        :param float y: Y position in km
        :param float z: Z position in km
        :param float vx: X-direction velocity in km/s
        :param float vy: Y-direction velocity in km/s
        :param float vz: Z-direction velocity in km/s
        '''

        kep = ['a', 'e', 'i', 'raan', 'aop', 'mu0']
        cart = ['x', 'y', 'z', 'vx', 'vy', 'vz']

        kep_orb = False
        cart_orb = False

        for key,val in kwargs.items():
            if key in kep:
                kep_orb = True
                setattr(self, key, val)

            elif key in cart:
                cart_orb = True
                setattr(self, key, val)

            else:
                raise TypeError('{} variable cannot be updated'.format(key))

        if kep_orb and cart_orb:
            raise TypeError('Cannot update both Cartesian and Kepler elements at the same time')

        if kep_orb:
            orb = n.array([self.a*1e3, self.e, self.i, self.aop, self.raan, dpt.mean2true(self.mu0,self.e,radians=False)],dtype=n.float)
            state_c = dpt.kep2cart(orb, m=n.array([self.m]),M_cent=self.M_cent, radians=False)*1e-3

        if cart_orb:
            state = n.array([self.x,self.y,self.z,self.vx,self.vy,self.vz],dtype=n.float)*1e3
            self.state_cart = state
            orb_c = dpt.cart2kep(state, m=self.m, M_cent=self.M_cent, radians=False)

        if kep_orb:
            self.x = state_c[0]
            self.y = state_c[1]
            self.z = state_c[2]
            self.vx = state_c[3]
            self.vy = state_c[4]
            self.vz = state_c[5]
            self.state_cart = state_c*1e3

        if cart_orb:
            self.a = orb_c[0]*1e-3
            self.e = orb_c[1]
            self.i = orb_c[2]
            self.raan = orb_c[4]
            self.aop = orb_c[3]
            self.mu0 = dpt.true2mean(orb_c[5],orb_c[1],radians=False)

    def __str__(self):
        p = '\nSpace object {} at epoch {} MJD:\n'.format(self.oid,self.mjd0)
        p+= 'CARTESIAN: x = {:.3f} km, y = {:.3f} km, z = {:.3f} km\nvx = {:.3f} km/s, vy = {:.3f} km/s, vz = {:.3f} km/s\n'.format( \
            self.x,self.y,self.z,self.vx,self.vy,self.vz)
        p+= 'ORBIT: \nsemi-major axis = {:.3f} km \neccentricity = {:.3f} \ninclination = {:.3f} deg \nperiapsis argument = {:.3f} deg \nascneding node = {:.3f} deg \nmean anomaly = {:.3f} deg\n'.format( \
            self.a, self.e, self.i, self.aop, self.raan, self.mu0)
        p+= 'PARAMETERS: diameter = {:.3f} m, drag coefficient = {:.3f}, albedo = {:.3f}, area = {:.3f}, mass = {:.3f} kg\n'.format(self.d,self.C_D,self.C_R,self.A,self.m)
        p+= 'ADDITIONAL KEYWORDS: ' + ', '.join([ '{} = {}'.format(key,val) for key,val in self.kwargs.items() ])
        return p


    def __enter__(self):
        return self

    def get_orbit(self,t):
        '''Gets ECEF position at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of positions as a function of time.
        :rtype: numpy.ndarray of size (3,len(t))
        '''
        ecefs = self.get_state(t)
        return(ecefs[:3,:])

    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def get_state(self,t):
        '''Gets ECEF state at specified times using propagator instance.

        :param float/list/numpy.ndarray t: Time relative epoch in seconds.

        :return: Array of state (position and velocity) as a function of time.
        :rtype: numpy.ndarray of size (6,len(t))
        '''
        if not isinstance(t,n.ndarray):
            if not isinstance(t,list):
                t = [t]
            t = n.array(t,dtype=n.float)
        
        #print('JD: {} \n T: {}'.format(dpt.mjd_to_jd(self.mjd0), t))
        
        ecefs = self.prop.get_orbit(
            t,
            a=self.a*1e3,
            e=self.e,
            inc=self.i,
            raan=self.raan,
            aop=self.aop,
            mu0=self.mu0,
            mjd0=self.mjd0,
            C_D=self.C_D,
            C_R=self.C_R,
            A=self.A,
            m=self.m,
            **self.kwargs
        )
        return ecefs


    def write_oem(self, t0, t1, n_items, fname):
        '''Writes OEM format file of orbital state for specific time interval.
        The states are linearly spaced in the specified time interval.

        :param float t0: Start time.
        :param float t1: End time.
        :param int n_items: State points between start and end times.
        '''
        tv=n.linspace(t0, t1, num=n_items, dtype=n.float)
        
        state=self.get_state(tv)
        
        ut0=dpt.jd_to_unix(dpt.mjd_to_jd(self.mjd0))
        
        ccsds_write.write_oem(tv+ut0, state, oid=self.oid, fname=fname)
        
        
if __name__ == "__main__":
    # test propagation

    o=SpaceObject(a=7000, e=0.0, i=69, raan=0, aop=0, mu0=0, mjd0=57125.7729)

    ecefs=o.get_state([0.0])
    
    o2 = SpaceObject.cartesian(ecefs[0], ecefs[1], ecefs[2], ecefs[3], ecefs[4], ecefs[5])
    
    ecefsc=o.get_state([0.0])

    print(ecefs-ecefsc)
    

    t=n.linspace(0,24*3600,num=1000, dtype=n.float)
    ecefs=o.get_state(t)

    from propagator_orekit import PropagatorOrekit

    o3 = SpaceObject(
        a=7000, e=0.0, i=69,
        raan=0, aop=0, mu0=0,
        C_D=2.3, A=1.0, m=1.0,
        C_R=1.0, oid=42,
        mjd0=57125.7729,
        propagator = PropagatorOrekit,
        propagator_options = {
            'in_frame': 'TEME',
            'out_frame': 'ITRF',
        },
    )

    ecefs2=o3.get_state(t)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],'-',alpha=0.5,color="black",label='SGP4')
    ax.plot(ecefs2[0,:],ecefs2[1,:],ecefs2[2,:],'-',alpha=0.5,color="green",label='Orekit')
    plt.title("Orbital propagation test")
    plt.legend()

    plt.show()
