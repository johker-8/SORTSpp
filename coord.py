"""Collection of common coordinate transformations.

   Some bits and pieces are from PySatel, but changed to work with numpy.
   There was a bug in geodetic2ecef in PySatel that is fixed. The other functions
   are implemented using information from wikipedia.

   Juha Vierinen 2013.
   Daniel Kastinen 2019: Bug-fixes and updates
"""
import os
from datetime import datetime, timedelta
from time import mktime
from numpy import power, degrees, radians, mat, cos, sin, arctan, sqrt, pi, arctan2, array, transpose, dot, arccos, sign
import math
import numpy

def cbrt(x):
    if x >= 0:
        return power(x, 1.0/3.0)
    else:
        return -power(abs(x), 1.0/3.0)

# Constants defined by the World Geodetic System 1984 (WGS84)
a = 6378.137*1e3
b = 6356.7523142*1e3
esq = 6.69437999014 * 0.001
e1sq = 6.73949674228 * 0.001
f = 1 / 298.257223563

def geodetic2ecef(lat, lon, alt):
    """
    Convert geodetic coordinates to ECEF.
    @lat, @lon in decimal degrees
    @alt in meters

    Uses WGS84.
    """
    lat, lon = radians(lat), radians(lon)
    xi = sqrt(1 - esq * sin(lat)**2)
    x = (a / xi + alt) * cos(lat) * cos(lon)
    y = (a / xi + alt) * cos(lat) * sin(lon)
    z = (a / xi * (1 - esq) + alt) * sin(lat)
    return(numpy.array([x, y, z]))

def enu2ecef(lat, lon, alt, n, e, u):
    """NEU (north/east/up) to ECEF coordinate system conversion. Degrees are used."""
    return ned2ecef(lat, lon, alt, n, e, -u)

def ned2ecef(lat, lon, alt, n, e, d):
    """NED (north/east/down) to ECEF coordinate system conversion. Degrees are used."""
    lat, lon = radians(lat), radians(lon)
    mx = array([[-sin(lon), -sin(lat) * cos(lon), cos(lat) * cos(lon)],
                [cos(lon), -sin(lat) * sin(lon), cos(lat) * sin(lon)],
                [0, cos(lat), sin(lat)]])
    enu = array([e, n, -1.0*d])
    res = dot(mx,enu)
    return res

def ecef2local(lat, lon, alt, x, y, z):
    """NED (east,north,up) from ECEF coordinate system conversion."""
    lat, lon = radians(lat), radians(lon)
    mx = array([[-sin(lon), -sin(lat) * cos(lon), cos(lat) * cos(lon)],
                [cos(lon), -sin(lat) * sin(lon), cos(lat) * sin(lon)],
                [0, cos(lat), sin(lat)]])
    enu = array([x, y, z])
    res = dot(numpy.linalg.inv(mx),enu)
    return(res)

def azel_ecef(lat, lon, alt, az, el):
    """Radar pointing (az,el) degrees to unit vector in ECEF."""
    return(ned2ecef(lat,lon,alt,
                    cos(-radians(az))*cos(radians(el)),
                    -sin(-radians(az))*cos(radians(el)),
                    -sin(radians(el))))

def cart_to_azel(vec):
    """Convert from Cartesian coordinates to spherical in a degrees east of north and elevation fashion

    """
    x = vec[0]
    y = vec[1]
    z = vec[2]
    r_ = sqrt(x**2 + y**2)
    if r_ < 1e-9:
        el = sign(z)*numpy.pi*0.5
        az = 0.0
    else:
        el = arctan(z/r_)
        az = math.pi/2 - arctan2(y,x)
    return degrees(az), degrees(el), sqrt(x**2 + y**2 +z**2)

def azel_to_cart(az, el, r):
    """Convert from spherical coordinates to Cartesian in a degrees east of north and elevation fashion

    """
    _az = radians(az)
    _el = radians(el)
    return r*array([sin(_az)*cos(_el), cos(_az)*cos(_el), sin(_el)])


def ecef2geodetic(x, y, z):
    """Convert ECEF coordinates to geodetic.
    J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates
    to geodetic coordinates," IEEE Transactions on Aerospace and
    Electronic Systems, vol. 30, pp. 957-961, 1994.
    
    According to WGS84.
    """
    r = sqrt(x * x + y * y)
    if r < 1e-9:
        h = abs(z) - b
        lat = sign(z)*numpy.pi/2
        lon = 0.0
    else:
        Esq = a * a - b * b
        F = 54 * b * b * z * z
        G = r * r + (1 - esq) * z * z - esq * Esq
        C = (esq * esq * F * r * r) / (pow(G, 3))
        S = cbrt(1 + C + sqrt(C * C + 2 * C))
        P = F / (3 * pow((S + 1 / S + 1), 2) * G * G)
        Q = sqrt(1 + 2 * esq * esq * P)
        r_0 =  -(P * esq * r) / (1 + Q) + sqrt(0.5 * a * a*(1 + 1.0 / Q) - \
            P * (1 - esq) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r)
        U = sqrt(pow((r - esq * r_0), 2) + z * z)
        V = sqrt(pow((r - esq * r_0), 2) + (1 - esq) * z * z)
        Z_0 = b * b * z / (a * V)
        h = U * (1 - b * b / (a * V))
        lat = arctan((z + e1sq * Z_0) / r)
        lon = arctan2(y, x)
    return(array([degrees(lat), degrees(lon), h]))

def geodetic_to_az_el_r(obs_lat, obs_lon, obs_h, target_lat, target_lon, target_h):
    """ When given a observer lat,long,h and target lat,long,h, provide azimuth, elevation, and range to target """
    up = ned2ecef(obs_lat, obs_lon, obs_h, 0.0, 0.0, -1.0)
    north = ned2ecef(obs_lat, obs_lon, obs_h, 1.0, 0.0, 0.0)
    east = ned2ecef(obs_lat, obs_lon, obs_h, 0.0, 1.0, 0.0)
    obs = array(geodetic2ecef(obs_lat, obs_lon, obs_h))
    target = array(geodetic2ecef(target_lat, target_lon, target_h))
    p_vec = (target-obs)
    az_p = dot(p_vec,north)*north+dot(p_vec,east)*east
    azs = sign(dot(p_vec,east))
    
    elevation = 90.0-180.0*arccos(dot(p_vec,up)/(sqrt(dot(p_vec,p_vec))*sqrt(dot(up,up))))/math.pi
    tmp_ang = dot(az_p,north)/(sqrt(dot(az_p,az_p))*sqrt(dot(north,north)))
    if tmp_ang > 1.0:
        tmp_ang = 1.0
    azimuth = azs*180.0*arccos(tmp_ang)/math.pi
    target_range = sqrt(dot(p_vec,p_vec))

    return(array([azimuth, elevation, target_range]))

def az_el_r2geodetic(obs_lat, obs_lon, obs_h, az, el, r):
    """ When given a observer lat,long,h and az,el and r, return lat,long,h of target """
    x = geodetic2ecef(obs_lat, obs_lon, obs_h) + azel_ecef(obs_lat, obs_lon, obs_h, az, el)*r
    llh = ecef2geodetic(x[0],x[1],x[2])
    if(llh[1] < 0.0):
        llh[1] = llh[1]+360.0
    return(llh)



def angle_deg(a,b):
    '''Angle in degrees between two vectors.
    
    :param numpy.ndarray a: Vector a
    :param numpy.ndarray a: Vector b
    :return: Angle in degrees between vectors a and b
    :rtype: float
    '''
    proj = numpy.dot(a,b)/(numpy.sqrt(numpy.dot(a,a))*numpy.sqrt(numpy.dot(b,b)))
    if proj > 1.0:
        proj = 1.0
    elif proj < -1.0:
        proj = -1.0
    return 180.0*numpy.arccos(proj)/numpy.pi

    
if __name__ == "__main__":
    east_ecef=azel_ecef(69.0, 19.0, 0.0, 90.0, 70.0)
    print(east_ecef)
    ecef_0 = geodetic2ecef(69.0, 19.0, 1.0)
    llh_0=ecef2geodetic(ecef_0[0],ecef_0[1],ecef_0[2])
    ep=ecef_0 + east_ecef*100e3
    llh_east=ecef2geodetic(ep[0],ep[1],ep[2])
    print(llh_0)
    print(llh_east)
#    test_coord()
