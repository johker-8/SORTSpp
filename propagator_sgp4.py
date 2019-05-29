#!/usr/bin/env python

'''SGP4 interface with SORTS++.

    Written in 2018 by Juha Vierinen
    based on code from Jan Siminski, ESA.
    Modified by Daniel Kastinen 2018/2019
'''

import sgp4.earth_gravity
import sgp4.io
import sgp4.propagation
import sgp4.model

import TLE_tools as tle
import dpt_tools as dpt

import numpy as np
import scipy.constants as consts

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from propagator_base import PropagatorBase


def gstime(jdut1):
    '''This function finds the greenwich sidereal time (iau-82).

    *References:* Vallado 2007, 193, Eq 3-43

    Author: David Vallado 719-573-2600    7 jun 2002
    Adapted to Python, Daniel Kastinen 2018

    :param float jdut1: Julian date of ut1 in days from 4713 bc

    :return: Greenwich sidereal time in radians, 0 to :math:`2\pi`
    :rtype: float
    '''
    twopi      = 2.0*np.pi;
    deg2rad    = np.pi/180.0;

    # ------------------------  implementation   ------------------
    tut1= ( jdut1 - 2451545.0 ) / 36525.0

    temp = - 6.2e-6 * np.multiply(np.multiply(tut1,tut1),tut1) + 0.093104 * np.multiply(tut1,tut1) \
           + (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841

    # 360/86400 = 1/240, to deg, to rad
    temp = np.fmod( temp*deg2rad/240.0,twopi )

    # ------------------------ check quadrants --------------------
    temp = np.where(temp < 0.0, temp+twopi, temp)


    gst = temp
    return gst



def polarm ( xp, yp, ttt, opt ):
    '''This function calculates the transformation matrix that accounts for polar motion. both the 1980 and 2000 theories are handled. note that the rotation order is different between 1980 and 2000.

    *References:* Vallado 2004, 207-209, 211, 223-224.

    Author: David Vallado 719-573-2600   25 jun 2002.
    Adapted to Python, Daniel Kastinen 2018


    :param float xp: x-axis polar motion coefficient in radians
    :param float yp: y-axis polar motion coefficient in radians
    :param float ttt: Julian centuries of tt (00 theory only)
    :param str opt: Model for polar motion to use, options are '01', '02', '80'.

    :return: Transformation matrix for ECEF to PEF
    :rtype: numpy.ndarray (3x3 matrix)

    '''
    cosxp = np.cos(xp)
    sinxp = np.sin(xp)
    cosyp = np.cos(yp)
    sinyp = np.sin(yp)

    pm = np.zeros([3,3])

    if opt == '80':
        pm[0,0] =  cosxp
        pm[0,1] =  0.0
        pm[0,2] = -sinxp
        pm[1,0] =  sinxp * sinyp
        pm[1,1] =  cosyp
        pm[1,2] =  cosxp * sinyp
        pm[2,0] =  sinxp * cosyp
        pm[2,1] = -sinyp
        pm[2,2] =  cosxp * cosyp

    else:
        convrt = np.pi / (3600.0*180.0)
        # approximate sp value in rad
        sp = -47.0e-6 * ttt * convrt
        cossp = np.cos(sp)
        sinsp = np.sin(sp)

        # form the matrix
        pm[0,0] =  cosxp * cossp
        pm[0,1] = -cosyp * sinsp + sinyp * sinxp * cossp
        pm[0,2] = -sinyp * sinsp - cosyp * sinxp * cossp
        pm[1,0] =  cosxp * sinsp
        pm[1,1] =  cosyp * cossp + sinyp * sinxp * sinsp
        pm[1,2] =  sinyp * cossp - cosyp * sinxp * sinsp
        pm[2,0] =  sinxp
        pm[2,1] = -sinyp * cosxp
        pm[2,2] =  cosyp * cosxp
    return pm


def ecef2teme( t, p, v, mjd0=57084, xp=0.0, yp=0.0, model='80', lod=0.0015563):
    '''Reverse operation, developed by Daniel Kastinen 2019

    # TODO: Write proper docstring
    p, v and output are all in units of km and km/s, as for teme2ecef
    '''

    if not isinstance(t, np.ndarray):
        raise TypeError('type(t) = {} not supported'.format(type(t)))


    if t.size != p.shape[1]:
        raise Exception("t and p lengths conflicting, t {}, p {} ".format(t.size, p.shape[1]))


    if len(t.shape) == 1:
        try:
            t = t.reshape(1, t.size)
        except:
            print('Reshaping failed: t-{}, p-{}, v-{}'.format(
                t.shape, p.shape, v.shape,
            ))
            raise

    if p.shape[0] != 3 or v.shape[0] != 3:
        try:
            p = p.reshape(3, p.size)
            v = v.reshape(3, v.size)
        except:
            print('Reshaping failed: t-{}, p-{}, v-{}'.format(
                t.shape, p.shape, v.shape,
            ))
            raise



    if not isinstance(xp, np.ndarray) or not isinstance(yp, np.ndarray):
        if np.abs(xp) < 1e-10 and np.abs(yp) < 1e-10:
            model = 'no'
    else:
        if xp.size != t.size or yp.size != t.size:
            raise Exception("Polar motion lengths conflicting with time length, xp {}, yp {} ".format(xp.size, yp.size))
        if xp.shape != t.shape:
            xp = xp.reshape(1, t.size)
            yp = yp.reshape(1, t.size)

    #from MJD to J2000 relative JD to BC-relative JD to julian centuries, 2400000.5 - 2451545.0 = - 51544.5
    #a Julian year (symbol: a) is a unit of measurement of time defined as exactly 365.25 days of 86400 SI seconds each
    ttt = (mjd0 + t/86400.0 - 51544.5)/(365.25*100.0)

    #JDUT1 is actually in days from 4713 bc
    jdut1 = mjd0 + t/86400.0 + 2400000.5

    # ------------------------ find gmst --------------------------
    theta = gstime( jdut1 );

    #find omega from nutation theory
    omega=  125.04452222  + ( -6962890.5390 *ttt + 7.455 *ttt*ttt + 0.008 *ttt*ttt*ttt ) / 3600.0;
    omega= np.radians(np.fmod( omega, 360.0  ));

    # ------------------------ find mean ast ----------------------
    # teme does not include the geometric terms here
    # after 1997, kinematic terms apply
    gmstg = np.where(
        jdut1 > 2450449.5,
        theta + 0.00264*np.pi /(3600*180)*np.sin(omega) + 0.000063*np.pi /(3600*180)*np.sin(2.0 *omega),
        theta,
    )
    gmstg = np.fmod( gmstg, 2.0*np.pi  )

    recef = np.empty(p.shape, dtype=p.dtype)
    vecef = np.empty(v.shape, dtype=v.dtype)

    if model == 'no':
        recef = p
        vecef = v
    elif model == '80':
        if isinstance(xp, np.ndarray):
            for tind in range(t.size):
                pm = polarm(xp[0,tind], yp[0,tind], 0.0, model)
                pm = np.linalg.inv(pm)
                recef[:,tind] = np.dot(pm, p[:,tind])
                vecef[:,tind] = np.dot(pm, v[:,tind])
        else:
            pm = polarm(xp, yp, 0.0, model)
            pm = np.linalg.inv(pm)
            recef = np.dot(pm, p)
            vecef = np.dot(pm, v)
    else:
        if isinstance(xp, np.ndarray):
            for tind in range(ttt.size):
                pm = polarm(xp[0,tind], yp[0,tind], ttt[0,tind], model)
                pm = np.linalg.inv(pm)
                recef[:,tind] = np.dot(pm, p[:,tind])
                vecef[:,tind] = np.dot(pm, v[:,tind])
        else:
            for tind in range(ttt.size):
                pm = polarm(xp, yp, ttt[0,tind], model)
                pm = np.linalg.inv(pm)
                recef[:,tind] = np.dot(pm, rpef[:,tind])
                vecef[:,tind] = np.dot(pm, vpef[:,tind])




    thetasa    = 7.29211514670698e-05*(1.0  - lod/86400.0)
    omegaearth = np.array([0., 0., -thetasa])

    for ind in range(t.shape[1]):
        vecef[:,ind] -= np.cross(omegaearth, recef[:,ind])

    rpef=np.zeros(p.shape)
    vpef=np.zeros(v.shape)

    # direct calculation of rotation matrix with different rotation for each
    # column of p
    theta = -theta

    rpef[0,:]=np.multiply(np.cos(theta),recef[0,:])+np.multiply(np.sin(theta),recef[1,:])
    rpef[1,:]=-np.multiply(np.sin(theta),recef[0,:])+np.multiply(np.cos(theta),recef[1,:])
    rpef[2,:]=recef[2,:]

    vpef[0,:]=np.multiply(np.cos(theta),vecef[0,:])+np.multiply(np.sin(theta),vecef[1,:])
    vpef[1,:]=-np.multiply(np.sin(theta),vecef[0,:])+np.multiply(np.cos(theta),vecef[1,:])
    vpef[2,:]=vecef[2,:]


    teme = np.empty((6,t.size))
    teme[:3,:] = rpef
    teme[3:,:] = vpef

    return teme

def teme2ecef( t, p, v, mjd0=57084, xp=0.0, yp=0.0, model='80', lod=0.0015563):
    '''This function trsnforms a vector from the true equator mean equniox frame
    (teme), to an earth fixed (ITRF) frame.  the results take into account
    the effects of sidereal time, and polar motion.


    *References:* Vallado  2007, 219-228.

    Author: David Vallado 719-573-2600, 10 dec 2007.
    Adapted to Python, Daniel Kastinen 2018

    :param numpy.ndarray t: numpy vector row of seconds relative to :code:`mjd0`
    :param numpy.ndarray p: numpy matrix of TEME positions, Cartesian coordinates in km. Columns correspond to times in t, and rows to x, y and z coordinates respectively.
    :param numpy.ndarray v: numpy matrix of TEME velocities, Cartesian coordinates in km/s. Columns correspond to times in t, and rows to x, y and z coordinates respectively.
    :param float mjd0: Modified julian date epoch that t vector is relative to
    :param float xp: x-axis polar motion coefficient in radians
    :param float yp: y-axis polar motion coefficient in radians
    :param str model: The polar motion model used in transformation, options are '80' or '00', see David Vallado documentation for more info.
    :param float lod: Excess length of day in seconds

    :return: State vector of position and velocity in km and km/s.
    :rtype: numpy.ndarray (6-D vector)

    **Uses:**

       * :func:`propagator_sgp4.gstime`
       * :func:`propagator_sgp4.polarm`



     [recef,vecef,aecef] = teme2ecef  ( rteme,vteme,ateme,ttt,jdut1,lod,xp,yp );

    '''


    if not isinstance(t, np.ndarray):
        raise TypeError('type(t) = {} not supported'.format(type(t)))


    if t.size != p.shape[1]:
        raise Exception("t and p lengths conflicting, t {}, p {} ".format(t.size, p.shape[1]))


    if len(t.shape) == 1:
        try:
            t = t.reshape(1, t.size)
        except:
            print('Reshaping failed: t-{}, p-{}, v-{}'.format(
                t.shape, p.shape, v.shape,
            ))
            raise

    if p.shape[0] != 3 or v.shape[0] != 3:
        try:
            p = p.reshape(3, p.size)
            v = v.reshape(3, v.size)
        except:
            print('Reshaping failed: t-{}, p-{}, v-{}'.format(
                t.shape, p.shape, v.shape,
            ))
            raise


    if not isinstance(xp, np.ndarray) or not isinstance(yp, np.ndarray):
        if np.abs(xp) < 1e-10 and np.abs(yp) < 1e-10:
            model = 'no'
    else:
        if xp.size != t.size or yp.size != t.size:
            raise Exception("Polar motion lengths conflicting with time length, xp {}, yp {} ".format(xp.size, yp.size))
        if xp.shape != t.shape:
            xp = xp.reshape(1, t.size)
            yp = yp.reshape(1, t.size)

    #from MJD to J2000 relative JD to BC-relative JD to julian centuries, 2400000.5 - 2451545.0 = - 51544.5
    #a Julian year (symbol: a) is a unit of measurement of time defined as exactly 365.25 days of 86400 SI seconds each
    ttt = (mjd0 + t/86400.0 - 51544.5)/(365.25*100.0)

    #JDUT1 is actually in days from 4713 bc
    jdut1 = mjd0 + t/86400.0 + 2400000.5

    # ------------------------ find gmst --------------------------
    theta = gstime( jdut1 );

    #find omega from nutation theory
    omega=  125.04452222  + ( -6962890.5390 *ttt + 7.455 *ttt*ttt + 0.008 *ttt*ttt*ttt ) / 3600.0;
    omega= np.radians(np.fmod( omega, 360.0  ));

    # ------------------------ find mean ast ----------------------
    # teme does not include the geometric terms here
    # after 1997, kinematic terms apply
    gmstg = np.where(
        jdut1 > 2450449.5,
        theta + 0.00264*np.pi /(3600*180)*np.sin(omega) + 0.000063*np.pi /(3600*180)*np.sin(2.0 *omega),
        theta,
    )
    gmstg = np.fmod( gmstg, 2.0*np.pi  )

    rpef=np.zeros(p.shape)
    vpef=np.zeros(v.shape)

    # direct calculation of rotation matrix with different rotation for each
    # column of p
    rpef[0,:]=np.multiply(np.cos(theta),p[0,:])+np.multiply(np.sin(theta),p[1,:])
    rpef[1,:]=-np.multiply(np.sin(theta),p[0,:])+np.multiply(np.cos(theta),p[1,:])
    rpef[2,:]=p[2,:]

    vpef[0,:]=np.multiply(np.cos(theta),v[0,:])+np.multiply(np.sin(theta),v[1,:])
    vpef[1,:]=-np.multiply(np.sin(theta),v[0,:])+np.multiply(np.cos(theta),v[1,:])
    vpef[2,:]=v[2,:]

    thetasa    = 7.29211514670698e-05*(1.0  - lod/86400.0)
    omegaearth = np.array([0., 0., thetasa])

    for ind in range(t.shape[1]):
        vpef[:,ind] -= np.cross(omegaearth, rpef[:,ind])

    recef = np.empty(rpef.shape, dtype=rpef.dtype)
    vecef = np.empty(vpef.shape, dtype=vpef.dtype)

    if model == 'no':
        recef = rpef
        vecef = vpef
    elif model == '80':
        if isinstance(xp, np.ndarray):
            for tind in range(t.size):
                pm = polarm(xp[0,tind], yp[0,tind], 0.0, model)
                recef[:,tind] = np.dot(pm, rpef[:,tind])
                vecef[:,tind] = np.dot(pm, vpef[:,tind])
        else:
            pm = polarm(xp, yp, 0.0, model)
            recef = np.dot(pm, rpef)
            vecef = np.dot(pm, vpef)
    else:
        if isinstance(xp, np.ndarray):
            for tind in range(ttt.size):
                pm = polarm(xp[0,tind], yp[0,tind], ttt[0,tind], model)
                recef[:,tind] = np.dot(pm, rpef[:,tind])
                vecef[:,tind] = np.dot(pm, vpef[:,tind])
        else:
            for tind in range(ttt.size):
                pm = polarm(xp, yp, ttt[0,tind], model)
                recef[:,tind] = np.dot(pm, rpef[:,tind])
                vecef[:,tind] = np.dot(pm, vpef[:,tind])

    ecef = np.empty((6,t.size))
    ecef[:3,:] = recef
    ecef[3:,:] = vecef

    return ecef


class SGP4:
    """
    The SGP4 class acts as a wrapper around the sgp4 module
    uploaded by Brandon Rhodes (http://pypi.python.org/pypi/sgp4/).

        
    It converts orbital elements into the TLE-like 'satellite'-structure which
    is used by the module for the propagation.

    Notes:
        The class can be directly used for propagation. Alternatively,
        a simple propagator function is provided below.
    """


    # Geophysical constants (WGS 72 values) for notational convinience
    WGS     = sgp4.earth_gravity.wgs72     # Model used within SGP4
    R_EARTH = WGS.radiusearthkm            # Earth's radius [km]
    GM      = WGS.mu                       # Grav.coeff.[km^3/s^2]
    RHO0    = 2.461e-8                     # Density at q0[kg/m^3]
    Q0      = 120.0                        # Reference height [km]
    S0      = 78.0                         # Reference height [km]

    # Time constants

    MJD_0 = 2400000.5

    def __init__(self, mjd_epoch, mean_elements, B):
        """
        Initialize SGP4 object from mean orbital elements and
        ballistic coefficient

        Creates a sgp4.model.Satellite object for mean element propagation
        First all units are converted to the ones used for TLEs, then
        they are modified to the sgp4 module standard.
        
        Input
        -----
        mjd_epoch     : epoch as Modified Julian Date (MJD)
        mean_elements : [a0,e0,i0,raan0,aop0,M0]
        B             : Ballistic coefficient ( 0.5*C_D*A/m )
        
        Remarks
        -------
        
        This object is not usable for TLE generation, but only for propagation
        as internal variables are modified by sgp4init.
        """

        a0     = mean_elements[0]         # Semi-major (a') at epoch [km]
        e0     = mean_elements[1]         # Eccentricity at epoch
        i0     = mean_elements[2]         # Inclination at epoch
        raan0  = mean_elements[3]         # RA of the ascending node at epoch
        aop0   = mean_elements[4]         # Argument of perigee at epoch
        M0     = mean_elements[5]         # Mean anomaly at epoch
    
        # Compute ballistic coefficient
        bstar  = 0.5*B*SGP4.RHO0 # B* in [1/m]
        n0 = np.sqrt(SGP4.GM) / (a0**1.5)
        
        # Scaling
        n0    = n0*(86400.0/(2*np.pi))          # Convert to [rev/d]
        bstar = bstar*(SGP4.R_EARTH*1000.0)     # Convert from [1/m] to [1/R_EARTH]
    
        # Compute year and day of year
        d   = mjd_epoch - 16480.0               # Days since 1904 Jan 1.0
        y   = int(int(d) / 365.25)                # Number of years since 1904
        doy = d - int(365.25*y)                 # Day of year
        if (y%4==0):
            doy+=1.0
                    
        # Create Satellite object and fill member variables
        sat = sgp4.model.Satellite()
        #Unique satellite number given in the TLE file.
        sat.satnum = 12345
        #Full four-digit year of this element set's epoch moment.
        sat.epochyr = 1904+y
        #Fractional days into the year of the epoch moment.
        sat.epochdays = doy
        #Julian date of the epoch (computed from epochyr and epochdays).
        sat.jdsatepoch = mjd_epoch + SGP4.MJD_0
        
        #First time derivative of the mean motion (ignored by SGP4).
        #sat.ndot
        #Second time derivative of the mean motion (ignored by SGP4).
        #sat.nddot
        #Ballistic drag coefficient B* in inverse earth radii.
        sat.bstar = bstar
        #Inclination in radians.
        sat.inclo = i0
        #Right ascension of ascending node in radians.
        sat.nodeo = raan0
        #Eccentricity.
        sat.ecco = e0
        #Argument of perigee in radians.
        sat.argpo = aop0
        #Mean anomaly in radians.
        sat.mo = M0
        #Mean motion in radians per minute.
        sat.no = n0 / ( 1440.0 / (2.0 *np.pi) )
        #
        sat.whichconst = SGP4.WGS
        
        sat.a = pow( sat.no*SGP4.WGS.tumin , (-2.0/3.0) )
        sat.alta = sat.a*(1.0 + sat.ecco) - 1.0
        sat.altp = sat.a*(1.0 - sat.ecco) - 1.0
    
        sgp4.propagation.sgp4init(SGP4.WGS, 'i', \
            sat.satnum, sat.jdsatepoch-2433281.5, sat.bstar,\
            sat.ecco, sat.argpo, sat.inclo, sat.mo, sat.no,\
            sat.nodeo, sat)

        # Store satellite object and epoch
        self.sat       = sat
        self.mjd_epoch = mjd_epoch
        
    def state(self, mjd):
        """
        Inertial position and velocity ([m], [m/s]) at epoch mjd
        

        :param float mjd: epoch where satellite should be propagated to
        
        """
        # minutes since reference epoch
        m = (mjd - self.mjd_epoch) * 1440.0
        r,v = sgp4.propagation.sgp4(self.sat, m)
        # convert to m and m/s ---- WHY IS THIS COMMENT HERE???
        return np.hstack((np.array(r),np.array(v)))
        
    def position(self, mjd):
        """
        Inertial position at epoch mjd
        
        :param float mjd: epoch where satellite should be propagated to
        """
        return self.state(mjd)[0:3]
        
    def velocity(self, mjd):
        """
        Inertial velocity at epoch mjd
                
        :param float mjd: epoch where satellite should be propagated to
        """
        return self.state(mjd)[3:7]


M_earth = SGP4.GM*1e9/consts.G
'''float: Mass of the Earth using the WGS72 convention.
'''

MU_earth = SGP4.GM*1e9
'''float: Standard gravitational parameter of the Earth using the WGS72 convention.
'''


def sgp4_propagation( mjd_epoch, mean_elements, B=0.0, dt=0.0, method=None):
    """
    Lazy SGP4 propagation using SGP4 class
    
    Create a satellite object from mean elements and propagate it

    :param list/numpy.ndarray mean_elements : [a0,e0,i0,raan0,aop0,M0]
    :param float B: Ballistic coefficient ( 0.5*C_D*A/m )
    :param float dt: Time difference w.r.t. element epoch in seconds
    :param float mjd_epoch: Epoch of elements as Modified Julian Date (MJD) Can be ignored if the exact epoch is unimportant.
    :param str method: Forces use of SGP4 or SDP4 depending on string 'n' or 'd'
    
    """
    mjd_ = mjd_epoch + dt / 86400.0
    obj = SGP4(mjd_epoch, mean_elements, B)
    if method is not None and method in ['n', 'd']:
        obj.sat.method = method
    return obj.state(mjd_)
    


class PropagatorSGP4(PropagatorBase):
    '''Propagator class implementing the SGP4 propagator.

    :ivar bool polar_motion: Determines if polar motion should be used in calculating ITRF frame.
    :ivar str polar_motion_model: String identifying the polar motion model to be used. Options are '80' or '00'.
    :ivar str out_frame: String identifying the output frame. Options are 'ITRF' or 'TEME'.

    The constructor creates a propagator instance with supplied options.

    :param bool polar_motion: Determines if polar motion should be used in calculating ITRF frame.
    :param str polar_motion_model: String identifying the polar motion model to be used. Options are '80' or '00'.
    :param str out_frame: String identifying the output frame. Options are 'ITRF' or 'TEME'.
    '''

    def __init__(self, polar_motion=False, polar_motion_model='80', out_frame='ITRF'):
        super(PropagatorSGP4, self).__init__()

        self.polar_motion = polar_motion
        self.polar_motion_model = polar_motion_model
        self.out_frame = out_frame


    def get_orbit(self, t, a, e, inc, raan, aop, mu0, mjd0, **kwargs):
        '''
        **Implementation:**
    
        All state-vector units are in meters.

        Keyword arguments contain only information needed for ballistic coefficient :code:`B` used by SGP4. Either :code:`B` or :code:`C_D`, :code:`A` and :code:`m` must be supplied.
        They also contain a option to give angles in radians or degrees. By default input is assumed to be degrees.

        **Frame:**

        The input frame is ECI (TEME) for orbital elements and Cartesian. The output frame is as standard ECEF (ITRF). But can be set to TEME.

        :param float B: Ballistic coefficient
        :param float C_D: Drag coefficient
        :param float A: Cross-sectional Area
        :param float m: Mass
        :param bool radians: If true, all angles are assumed to be in radians.
        '''
        if 'm' in kwargs:
            m = kwargs['m']
        else:
            m = 0.0

        if 'radians' in kwargs:
            radians = kwargs['radians']
        else:
            radians = False
        
        if not radians:
            inc = np.radians(inc)
            mu0 = np.radians(mu0)
            raan = np.radians(raan)
            aop = np.radians(aop)

        orb = np.array([a, e, inc, aop, raan, dpt.mean2true(mu0, e, radians=True)],dtype=np.float)

        state_c = dpt.kep2cart(orb, m=m, M_cent=M_earth, radians=True)

        return self.get_orbit_cart(t, x=state_c[0], y=state_c[1], z=state_c[2], vx=state_c[3], vy=state_c[4], vz=state_c[5], mjd0=mjd0, **kwargs)


    def get_orbit_TLE(self, t, line1, line2):
        '''Takes a TLE and propagates it forward in time directly using the SGP4 algorithm.

        :param float/list/numpy.ndarray t: Time in seconds to propagate relative the initial state epoch.
        :param str line1: TLE line 1
        :param str line2: TLE line 2

        '''
        jd0 = tle.tle_jd(line1)
        mjd0 = dpt.jd_to_mjd(jd0)

        jddates = jd0 + t/(3600.0*24.0)

        states_TEME = tle.TLE_propagation_TEME(line1, line2, jddates)

        if self.out_frame == 'TEME':
            return states_TEME
            
        elif self.out_frame == 'ITRF':
            ecefs = np.empty(states_TEME.shape, dtype=states_TEME.dtype)

            if self.polar_motion:
                PM_data = tle.get_Polar_Motion(jddates)
                for jdi in range(jddates.size):
                    ecefs[:,jdi] = tle.TEME_to_ITRF(
                        states_TEME[:,jdi],
                        jddates[jdi],
                        PM_data[jdi,0],
                        PM_data[jdi,1],
                    )
            else:
                for jdi in range(jddates.size):
                    ecefs[:,jdi] = tle.TEME_to_ITRF(
                        states_TEME[:,jdi],
                        jddates[jdi],
                        0.0,
                        0.0,
                    )
            return ecefs

        else:
            raise Exception('Output frame {} not found'.format(self.out_frame))



    def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
        '''
        **Implementation:**

        All state-vector units are in meters.

        Keyword arguments contain only information needed for ballistic coefficient :code:`B` used by SGP4. Either :code:`B` or :code:`C_D`, :code:`A` and :code:`m` must be supplied.
        They also contain a option to give angles in radians or degrees. By default input is assumed to be degrees.

        **Frame:**

        The input frame is ECI (TEME) for orbital elements and Cartesian. The output frame is always ECEF.

        :param float B: Ballistic coefficient
        :param float C_D: Drag coefficient
        :param float A: Cross-sectional Area
        :param float m: Mass
        :param bool radians: If true, all angles are assumed to be in radians.
        '''
        t = self._make_numpy(t)

        if 'B' in kwargs:
            B = kwargs['B']
        else:
            B = 0.5*kwargs['C_D']*kwargs['A']/kwargs['m']

        state_c = np.array([x,y,z,vx,vy,vz], dtype=np.float)
        state_c *= 1e-3 #to km

        mean_elements = tle.TEME_to_TLE(state_c, mjd0=mjd0, kepler=False)

        if np.any(np.isnan(mean_elements)):
            raise Exception('Could not compute SGP4 initial state: {}'.format(mean_elements))

        # Create own SGP4 object
        obj = SGP4(mjd0, mean_elements, B)

        mjdates = mjd0 + t/86400.0
        pos=np.zeros([3,t.size])
        vel=np.zeros([3,t.size])

        for mi,mjd in enumerate(mjdates):
            y = obj.state(mjd)
            pos[:,mi] = y[:3]
            vel[:,mi] = y[3:]

        if self.out_frame == 'TEME':
            states=np.empty((6,t.size), dtype=np.float)
            states[:3,:] = pos*1e3
            states[3:,:] = vel*1e3
            return states

        elif self.out_frame == 'ITRF':
            if self.polar_motion:
                PM_data = tle.get_Polar_Motion(dpt.mjd_to_jd(mjdates))
                xp = PM_data[:,0]
                xp.shape = (1,xp.size)
                yp = PM_data[:,1]
                yp.shape = (1,yp.size)
            else:
                xp = 0.0
                yp = 0.0

            ecefs = teme2ecef(t, pos, vel, mjd0=mjd0, xp=xp, yp=yp ,model=self.polar_motion_model)
            ecefs *= 1e3 #to meter
            return ecefs
        else:
            raise Exception('Output frame {} not found'.format(self.out_frame))


    

class PropagatorTLE(PropagatorSGP4):
    def __init__(self, polar_motion=False, polar_motion_model='80', out_frame='ITRF'):
        super(PropagatorTLE, self).__init__(polar_motion=polar_motion, polar_motion_model=polar_motion_model, out_frame=out_frame)


    def get_orbit(self, t, a, e, inc, raan, aop, mu0, mjd0, **kwargs):
        '''
        **Implementation:**

        Direct propagation of TLE. All input elements are ignored except for the two lines in :code:`**kwargs`.

        **Frame:**
            
            The output frame is (ITRF) ECEF or (TEME) ECI.

        :param str line1: TLE line 1
        :param str line2: TLE line 2
        '''

        line1 = kwargs['line1']
        line2 = kwargs['line2']
        return self.get_orbit_TLE(t, line1, line2)

    def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
        '''
        **Implementation:**

        Direct propagation of TLE. All input elements are ignored except for the two lines in :code:`**kwargs`.

        **Frame:**
            
            The output frame is (ITRF) ECEF or (TEME) ECI.

        :param str line1: TLE line 1
        :param str line2: TLE line 2
        '''

        line1 = kwargs['line1']
        line2 = kwargs['line2']
        return self.get_orbit_TLE(t, line1, line2)


if __name__ == "__main__":
    import time
    from propagator_base import plot_orbit_3d


    prop = PropagatorSGP4()

    mjd0 = dpt.jd_to_mjd(2457126.2729)
    C_D = 2.3
    m = 8000
    A = 1.0

    t = np.arange(0,24*3600, dtype=np.float)
    
    t0=time.time()

    ecefs = prop.get_orbit(
        t=t, mjd0=mjd0,
        a=7000e3, e=0.0, inc=90.0,
        raan=10, aop=10, mu0=40.0,
        C_D=C_D, m=m, A=A,
    )

    t1=time.time()

    print('exec time: {} sec'.format(t1-t0))
    
    plot_orbit_3d(ecefs)

    prop = PropagatorSGP4(polar_motion=True, polar_motion_model='00')
    
    t0=time.time()

    ecefs = prop.get_orbit(
        t=t, mjd0=mjd0,
        a=7000e3, e=0.0, inc=90.0,
        raan=10, aop=10, mu0=40.0,
        C_D=C_D, m=m, A=A,
    )

    t1=time.time()

    print('exec time: {} sec'.format(t1-t0))

    plot_orbit_3d(ecefs)
