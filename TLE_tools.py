'''Collection of useful functions for handling TLE's.


**Links:**
  * `AIAA 2006-6753 <https://celestrak.com/publications/AIAA/2006-6753/>`_
  * `python-skyfield <https://github.com/skyfielders/python-skyfield>`_


'''

import os
import time
import pdb

import numpy as np
from sgp4.io import twoline2rv
import dpt_tools as dpt


from scipy.optimize import minimize

try:
    import propagator_sgp4
except:
    propagator_sgp4 = None


def TLE_propagation_TEME(line1, line2, jd_ut1, wgs = '72'):
    '''Convert Two-line element to TEME coordinates at a specific Julian date.
    
    :param str line1: TLE line 1
    :param str line2: TLE line 2
    :param float/numpy.ndarray jd_ut1: Julian Date UT1 to propagate TLE to.
    :param str wgs: The used WGS standard, options are :code:`'72'` or :code:`'84'`.

    :return: (6,len(jd_ut1)) numpy.ndarray of Cartesian states [SI units]
    '''

    if wgs == '72':
        from sgp4.earth_gravity import wgs72 as wgs_sys
    elif wgs == '84':
        from sgp4.earth_gravity import wgs84 as wgs_sys
    else:
        raise Exception('WGS standard "{}" not recognized.'.format(wgs))

    satellite = twoline2rv(line1, line2, wgs_sys)

    if isinstance(jd_ut1, np.ndarray):
        if len(jd_ut1.shape) > 1:
            raise Exception('Only 1-D date array allowed: jd_ut1.shape = {}'.format(jd_ut1.shape))
    else:
        if isinstance(jd_ut1, float):
            jd_ut1 = np.array([jd_ut1], dtype=np.float)

    states = np.empty((6, jd_ut1.size), dtype=np.float)

    for jdi in range(jd_ut1.size):
        year, month, day, hour, minute, second, usec = dpt.npdt2date(dpt.mjd2npdt(dpt.jd_to_mjd(jd_ut1[jdi])))

        position, velocity = satellite.propagate(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second + usec*1e-6,
        )

        state = np.array(position + velocity, dtype=np.float)
        state *= 1e3 #km to m

        states[:,jdi] = state

    return states

def tle_npdt(line1):
    year = int(line1[18:20])

    if year < 60:
        year = 2000 + year
    else:
        year = 1900 + year

    yearday_frac = float(line1[20:32]) - 1.0
    yearday_dt = np.timedelta64(int(yearday_frac*3600.0*24.0*1e6),'us')

    np_date = np.datetime64('{}-01-01T00:00'.format(year), 'us') + yearday_dt
    return np_date


def tle_jd(line1):
    np_date = tle_npdt(line1)
    return dpt.mjd_to_jd(dpt.npdt2mjd(np_date))


def tle_date(line1):
    np_date = tle_npdt(line1)
    return dpt.npdt2date(np_date)


def TLE_to_TEME(line1, line2, wgs = '72'):
    '''Convert Two-line element to TEME coordinates and a Julian date epoch.

    Here it is assumed that the TEME frame uses:
    The Cartesian coordinates produced by the SGP4/SDP4 model have their z
    axis aligned with the true (instantaneous) North pole and the x axis
    aligned with the mean direction of the vernal equinox (accounting for
    precession but not nutation). This actually makes sense since the
    observations are collected from a network of sensors fixed to the
    earth's surface (and referenced to the true equator) but the position
    of the earth in inertial space (relative to the vernal equinox) must
    be estimated.
    
    :param str line1: TLE line 1
    :param str line2: TLE line 2
    :param str wgs: The used WGS standard, options are :code:`'72'` or :code:`'84'`.

    :return: tuple of (6-D numpy.ndarray Cartesian state [SI units], epoch in Julian Date UT1)
    '''

    year, month, day, hour, minute, second, usec = tle_date(line1)
    
    if wgs == '72':
        from sgp4.earth_gravity import wgs72 as wgs_sys
    elif wgs == '84':
        from sgp4.earth_gravity import wgs84 as wgs_sys
    else:
        raise Exception('WGS standard "{}" not recognized.'.format(wgs))

    satellite = twoline2rv(line1, line2, wgs_sys)
    position, velocity = satellite.propagate(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second,
    )

    state = np.array(position + velocity, dtype=np.float)
    state *= 1e3 #km to m

    return state, tle_jd(line1)


def theta_GMST1982(jd_ut1):
    """Return the angle of Greenwich Mean Standard Time 1982 given the JD.
    This angle defines the difference between the idiosyncratic True
    Equator Mean Equinox (TEME) frame of reference used by SGP4 and the
    more standard Pseudo Earth Fixed (PEF) frame of reference.


    *Reference:* AIAA 2006-6753 Appendix C.

    Original work Copyright (c) 2013-2018 Brandon Rhodes under the MIT license
    Modified work Copyright (c) 2019 Daniel Kastinen

    :param float jd_ut1: UT1 Julian date.
    :returns: tuple of (Earth rotation [rad], Earth angular velocity [rad/day])

    """
    _second = 1.0 / (24.0 * 60.0 * 60.0)

    T0 = 2451545.0

    t = (jd_ut1 - T0) / 36525.0
    g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t
    dg = 8640184.812866 + (0.093104 * 2.0 + (-6.2e-6 * 3.0) * t) * t
    theta = (jd_ut1 % 1.0 + g * _second % 1.0) * 2.0 * np.pi
    theta_dot = (1.0 + dg * _second / 36525.0) * 2.0 * np.pi
    return theta, theta_dot

#when imported, get the full real path for the data loading
try:
    _base_path = '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/'
except:
    _base_path = ''

def get_IERS_EOP(fname = _base_path + 'data/eopc04_IAU2000.62-now'):
    '''Loads the IERS EOP data into memory.

    Note: Column descriptions are hard-coded in the function and my change if standard IERS format is changed.

    :param str fname: path to input IERS data file.
    :return: tuple of (numpy.ndarray, list of column descriptions)

    '''
    data = np.genfromtxt(fname, skip_header=14)
    header = [
        'Date year (0h UTC)',
        'Date month (0h UTC)',
        'Date day (0h UTC)',
        'MJD',
        'x (arcseconds)',
        'y (arcseconds)',
        'UT1-UTC (s)',
        'LOD (s)',
        'dX (arcseconds)',
        'dY (arcseconds)',
        'x Err (arcseconds)',
        'y Err (arcseconds)',
        'UT1-UTC Err (s)',
        'LOD Err (s)',
        'dX Err (arcseconds)',
        'dY Err (arcseconds)',
    ]



    return data, header


_EOP_data, _EOP_header = get_IERS_EOP()

_mjd_const = 2400000.5
_arcseconds_to_rad = np.pi/(60.0*60.0*180.0)

def _get_jd_rows(jd_ut1):
    mjd = jd_ut1 - _mjd_const
    row = np.argwhere(np.abs(_EOP_data[:,3] - np.floor(mjd)) < 1e-1 )
    if len(row) == 0:
        raise Exception('No EOP data available for JD: {}'.format(jd_ut1))
    row = int(row)
    return _EOP_data[row:(row+2),:]

def _interp_rows(_jd_ut1, cols):
    if not isinstance(_jd_ut1, np.ndarray):
        if not isinstance(_jd_ut1, float):
            raise Exception('Only numpy.ndarray and float input allowed')

        _jd_ut1 = np.array([_jd_ut1], dtype=np.float)
    else:
        if max(_jd_ut1.shape) != _jd_ut1.size:
            raise Exception('Only 1D input arrays allowed')
        else:
            _jd_ut1 = _jd_ut1.copy().flatten()

    ret = np.empty((_jd_ut1.size, len(cols)), dtype=np.float)

    for jdi, jd_ut1 in enumerate(_jd_ut1):
        rows = _get_jd_rows(jd_ut1)
        frac = jd_ut1 - np.floor(jd_ut1)

        for ci, col in enumerate(cols):
            ret[jdi, ci] = rows[0,col]*(1.0 - frac) + rows[1,col]*frac

    return ret

def get_DUT(jd_ut1):
    '''Get the Difference UT between UT1 and UTC, :math:`DUT1 = UT1 - UTC`. This function interpolates between data given by IERS.
    
    :param float/numpy.ndarray jd_ut1: Input Julian date in UT1.
    :return: DUT
    :rtype: numpy.ndarray
    '''
    return _interp_rows(jd_ut1, [6])

def get_Polar_Motion(jd_ut1):
    '''Get the Polar motion coefficients :math:`x_p` and :math:`y_p` used in EOP. This function interpolates between data given by IERS.
    
    :param float/numpy.ndarray jd_ut1: Input Julian date in UT1.
    :return: :math:`x_p` as column 0 and :math:`y_p` as column 1
    :rtype: numpy.ndarray
    '''
    return _interp_rows(jd_ut1, [4,5])*_arcseconds_to_rad



def TEME_to_ITRF(TEME, jd_ut1, xp, yp):
    """Convert TEME position and velocity into standard ITRS coordinates.
    This converts a position and velocity vector in the idiosyncratic
    True Equator Mean Equinox (TEME) frame of reference used by the SGP4
    theory into vectors into the more standard ITRS frame of reference.

    *Reference:* AIAA 2006-6753 Appendix C.

    Original work Copyright (c) 2013-2018 Brandon Rhodes under the MIT license
    Modified work Copyright (c) 2019 Daniel Kastinen

    Since TEME uses the instantaneous North pole and mean direction
    of the Vernal equinox, a simple GMST and polar motion transformation will move to ITRS.

    # TODO: There is some ambiguity about if this is ITRS00 or something else? I dont know.

    :param numpy.ndarray TEME: 6-D state vector in TEME frame given in SI-units.
    :param float jd_ut1: UT1 Julian date.
    :param float xp: Polar motion constant for rotation around x axis
    :param float yp: Polar motion constant for rotation around y axis
    :return: ITRF 6-D state vector given in SI-units.
    :rtype: numpy.ndarray
    """

    if len(TEME.shape) > 1:
        rTEME = TEME[:3, :]
        vTEME = TEME[3:, :]*3600.0*24.0
    else:
        rTEME = TEME[:3]
        vTEME = TEME[3:]*3600.0*24.0

    theta, theta_dot = theta_GMST1982(jd_ut1)
    zero = theta_dot * 0.0
    angular_velocity = np.array([zero, zero, -theta_dot])
    R = dpt.rot_mat_z(-theta)

    if len(rTEME.shape) == 1:
        rPEF = (R).dot(rTEME)
        vPEF = (R).dot(vTEME) + np.cross(angular_velocity, rPEF)
    else:
        rPEF = np.einsum('ij...,j...->i...', R, rTEME)
        vPEF = np.einsum('ij...,j...->i...', R, vTEME) + np.cross(
            angular_velocity, rPEF, 0, 0).T

    if np.abs(xp) < 1e-10 and np.abs(yp) < 1e-10:
        rITRF = rPEF
        vITRF = vPEF
    else:
        W = (dpt.rot_mat_x(yp)).dot(dpt.rot_mat_y(xp))
        rITRF = (W).dot(rPEF)
        vITRF = (W).dot(vPEF)

    ITRF = np.empty(TEME.shape, dtype=TEME.dtype)
    if len(TEME.shape) > 1:
        ITRF[:3,:] = rITRF
        ITRF[3:,:] = vITRF/(3600.0*24.0)
    else:
        ITRF[:3] = rITRF
        ITRF[3:] = vITRF/(3600.0*24.0)

    return ITRF



def ITRF_to_TEME(ITRF, jd_ut1, xp, yp):
    """Convert ITRF position and velocity into the idiosyncratic
    True Equator Mean Equinox (TEME) frame of reference used by the SGP4
    theory.

    Modified work Copyright (c) 2019 Daniel Kastinen

    # TODO: There is some ambiguity about if this is ITRS00 or something else? I dont know.

    :param numpy.ndarray ITRF: 6-D state vector in ITRF frame given in SI-units.
    :param float jd_ut1: UT1 Julian date.
    :param float xp: Polar motion constant for rotation around x axis
    :param float yp: Polar motion constant for rotation around y axis
    :return: ITRF 6-D state vector given in SI-units.
    :rtype: numpy.ndarray
    """

    if len(ITRF.shape) > 1:
        rITRF = ITRF[:3, :]
        vITRF = ITRF[3:, :]*3600.0*24.0
    else:
        rITRF = ITRF[:3]
        vITRF = ITRF[3:]*3600.0*24.0

    theta, theta_dot = theta_GMST1982(jd_ut1)
    zero = theta_dot * 0.0
    angular_velocity = np.array([zero, zero, -theta_dot])
    R = dpt.rot_mat_z(theta)

    if np.abs(xp) < 1e-10 and np.abs(yp) < 1e-10:
        rPEF = rITRF
        vPEF = vITRF
    else:
        W = (dpt.rot_mat_y(-xp)).dot(dpt.rot_mat_x(-yp))
        rPEF = (W).dot(rITRF)
        vPEF = (W).dot(vITRF)

    if len(rITRF.shape) == 1:
        rTEME = (R).dot(rPEF)
        vTEME = (R).dot(vPEF - np.cross(angular_velocity, rPEF))
    else:
        rTEME = np.einsum('ij...,j...->i...', R, rPEF)
        vTEME = np.einsum('ij...,j...->i...', R, vPEF - np.cross(angular_velocity, rPEF, 0, 0) )

    TEME = np.empty(ITRF.shape, dtype=ITRF.dtype)
    if len(TEME.shape) > 1:
        TEME[:3,:] = rTEME
        TEME[3:,:] = vTEME/(3600.0*24.0)
    else:
        TEME[:3] = rTEME
        TEME[3:] = vTEME/(3600.0*24.0)

    return TEME


def _sgp4_elems2cart(kep, radians=True):
    '''Orbital elements to cartesian coordinates. Wrap DPT-function to use mean anomaly, km and reversed order on aoe and raan.
    
    Neglecting mass is sufficient for this calculation (the standard gravitational parameter is 24 orders larger then the change).
    '''
    _kep = kep.copy()
    _kep[0] *= 1e3
    tmp = _kep[4]
    _kep[4] = _kep[3]
    _kep[3] = tmp
    _kep[5] = dpt.mean2true(_kep[5], _kep[1], radians=radians)
    cart = dpt.kep2cart(kep, m=0.0, M_cent=propagator_sgp4.M_earth, radians=radians)
    return cart

def _cart2sgp4_elems(cart, radians=True):
    '''Cartesian coordinates to orbital elements. Wrap DPT-function to use mean anomaly, km and reversed order on aoe and raan.
    
    Neglecting mass is sufficient for this calculation (the standard gravitational parameter is 24 orders larger then the change).
    '''
    kep = dpt.cart2kep(cart*1e3, m=0.0, M_cent=propagator_sgp4.M_earth, radians=radians)
    kep[0] *= 1e-3
    tmp = kep[4]
    kep[4] = kep[3]
    kep[3] = tmp
    kep[5] = dpt.true2mean(kep[5], kep[1], radians=radians)
    return kep



def TEME_to_TLE_OPTIM(state, mjd0, kepler=False, tol=1e-6, tol_v=1e-7, method=None):
    '''Convert osculating orbital elements in TEME
    to mean elements used in two line element sets (TLE's).

    :param numpy.ndarray kep: Osculating State (position and velocity) vector in km and km/s, TEME frame. If :code:`kepler = True` then state is osculating orbital elements, in km and radians. Orbital elements are semi major axis (km), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
    :param bool kepler: Indicates if input state is kepler elements or cartesian.
    :param float mjd0: Modified Julian date for state, important for SDP4 iteration.
    :param float tol: Wanted precision in position of mean element conversion in km.
    :param float tol_v: Wanted precision in velocity mean element conversion in km/s.
    :param str method: Forces use of SGP4 or SDP4 depending on string 'n' or 'd', if None method is automatically chosen based on orbital period.
    :return: mean elements of: semi major axis (km), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
    :rtype: numpy.ndarray
    '''
    
    if kepler:
        state_cart = _sgp4_elems2cart(state, radians=True)
    else:
        state_cart = state
    
    init_elements = _cart2sgp4_elems(state_cart, radians=True)

    def find_mean_elems(mean_elements):
        # Mean elements and osculating state
        state_osc = propagator_sgp4.sgp4_propagation(mjd0, mean_elements, B=0.0, dt=0.0, method=method)

        # Correction of mean state vector
        d = state_cart - state_osc
        return np.linalg.norm(d*1e3)

    bounds = [(None, None), (0.0, 0.999), (0.0, np.pi)] + [(0.0, np.pi*2.0)]*3

    opt_res = minimize(find_mean_elems, init_elements,
        #method='powell',
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': np.sqrt(tol**2 + tol_v**2)}
    )
    mean_elements = opt_res.x

    return mean_elements



def TEME_to_TLE(state, mjd0, kepler=False, tol=1e-6, tol_v=1e-7):
    '''Convert osculating orbital elements in TEME
    to mean elements used in two line element sets (TLE's).

    :param numpy.ndarray kep: Osculating State (position and velocity) vector in km and km/s, TEME frame. If :code:`kepler = True` then state is osculating orbital elements, in km and radians. Orbital elements are semi major axis (km), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
    :param bool kepler: Indicates if input state is kepler elements or cartesian.
    :param float mjd0: Modified Julian date for state, important for SDP4 iteration.
    :param float tol: Wanted precision in position of mean element conversion in km.
    :param float tol_v: Wanted precision in velocity mean element conversion in km/s.
    :return: mean elements of: semi major axis (km), orbital eccentricity, orbital inclination (radians), right ascension of ascending node (radians), argument of perigee (radians), mean anomaly (radians)
    :rtype: numpy.ndarray
    '''
    
    if kepler:
        state_mean = _sgp4_elems2cart(state, radians=True)
        state_kep = state
        state_cart = state_mean.copy()
    else:
        state_mean = state.copy()
        state_cart = state
        state_kep = _cart2sgp4_elems(state, radians=True)

    #method        : Forces use of SGP4 or SDP4 depending on string 'n' or 'd', None is automatic method
    # Fix used model (SGP or SDP)
    T = 2.0*np.pi*np.sqrt(np.power(state_kep[0], 3) / propagator_sgp4.SGP4.GM)/60.0
    if T > 220.0:
        method = 'd'
    else:
        method = None

    iter_max = 300  # Maximum number of iterations

    # Iterative determination of mean elements
    for it in range(iter_max):
        # Mean elements and osculating state
        mean_elements = _cart2sgp4_elems(state_mean, radians=True)

        if it > 0 and mean_elements[1] > 1:
            #Assumptions of osculation within slope not working, go to general minimization algorithms
            mean_elements = TEME_to_TLE_OPTIM(state_cart, mjd0=mjd0, kepler=False, tol=tol, tol_v=tol_v, method=method)
            break

        state_osc = propagator_sgp4.sgp4_propagation(mjd0, mean_elements, B=0.0, dt=0.0, method=method)

        # Correction of mean state vector
        d = state_cart - state_osc
        state_mean += d
        if it > 0:
            dr_old = dr
            dv_old = dv

        dr = np.linalg.norm(d[:3])  # Position change
        dv = np.linalg.norm(d[3:])  # Velocity change

        if it > 0:
            if dr_old < dr or dv_old < dv:
                #Assumptions of osculation within slope not working, go to general minimization algorithms
                mean_elements = TEME_to_TLE_OPTIM(state_cart, mjd0=mjd0, kepler=False, tol=tol, tol_v=tol_v, method=method)
                break

        if dr < tol and dv < tol_v:   # Iterate until position changes by less than eps
            break
        if it == iter_max - 1:
            #Iterative method not working, go to general minimization algorithms
            mean_elements = TEME_to_TLE_OPTIM(state_cart, mjd0=mjd0, kepler=False, tol=tol, tol_v=tol_v, method=method)

    return mean_elements


def tle_id(line1):
    '''Extracts the Satellite number from the first line of a TLE.
    '''
    return line1[2:7]


def tle_bstar(line1):
    '''Extracts the BSTAR drag coefficient as a float from the first line of a TLE.
    '''
    bstar = float(line1[53:59])
    exp = float(line1[59:61].strip())
    return bstar*10.0**(exp)
