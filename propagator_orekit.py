#!/usr/bin/env python

'''Wrapper for the Orekit propagator into SORTS++ format.

**Links:**
    * `orekit <https://www.orekit.org/>`_
    * `orekit python <https://gitlab.orekit.org/orekit-labs/python-wrapper>`_
    * `orekit python guide <https://gitlab.orekit.org/orekit-labs/python-wrapper/wikis/Manual-Installation-of-Python-Wrapper>`_
    * `Hipparchus <https://www.hipparchus.org/>`_
    * `orekit 9.3 api <https://www.orekit.org/static/apidocs/index.html>`_
    * `JCC <https://pypi.org/project/JCC/>`_


**Example usage:**

Simple propagation showing time difference due to loading of model data.

.. code-block:: python

    from propagator_orekit import PropagatorOrekit
    import time
    import numpy as n
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    t0 = time.time()
    p = PropagatorOrekit()
    print('init time: {} sec'.format(time.time() - t0))

    init_data = {
        'a': 9000,
        'e': 0.0,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 8000,
        'A': 1.0,
    }
    t = n.linspace(0,24*3600.0, num=1000, dtype=n.float)

    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (first): {} sec'.format(time.time() - t0))


    t0 = time.time()
    ecefs = p.get_orbit(t, **init_data)
    print('get orbit time (second): {} sec'.format(time.time() - t0))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ecefs[0,:], ecefs[1,:], ecefs[2,:],".",color="green")
    plt.show()


Propagation using custom settings:

.. code-block:: python

    from propagator_orekit import PropagatorOrekit
    import numpy as n
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    init_data = {
        'a': 7500,
        'e': 0.1,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 8000,
        'A': 1.0,
    }
    t = n.linspace(0,10*3600.0, num=10000, dtype=n.float)

    
    p2 = PropagatorOrekit()
    print(p2)
    ecefs2 = p2.get_orbit(t, **init_data)

    p1 = PropagatorOrekit(earth_gravity='Newtonian', radiation_pressure=False, solarsystem_perturbers=[], drag_force=False)
    print(p1)
    ecefs1 = p1.get_orbit(t, **init_data)


    dr = n.sqrt(n.sum((ecefs1[:3,:] - ecefs2[:3,:])**2, axis=0))
    dv = n.sqrt(n.sum((ecefs1[3:,:] - ecefs2[3:,:])**2, axis=0))

    r1 = n.sqrt(n.sum(ecefs1[:3,:]**2, axis=0))
    r2 = n.sqrt(n.sum(ecefs1[:3,:]**2, axis=0))

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(311)
    ax.plot(t/3600.0, dr*1e-3)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Position difference [km]')
    ax.set_title('Propagation difference diameter simple vs advanced models')
    ax = fig.add_subplot(312)
    ax.plot(t/3600.0, dv*1e-3)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Velocity difference [km/s]')
    ax = fig.add_subplot(313)
    ax.plot(t/3600.0, r1*1e-3, color="green",label='Simple model')
    ax.plot(t/3600.0, r2*1e-3, color="red",label='Advanced model')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel('Distance from Earth center [km]')
    plt.legend()


    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],"-",alpha=0.5,color="green",label='Simple model')
    ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],"-",alpha=0.5,color="red",label='Advanced model')
    plt.legend()
    plt.show()


Propagation using different coordinate systems:

.. code-block:: python

    from propagator_orekit import PropagatorOrekit
    import numpy as n
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    p = PropagatorOrekit(in_frame='ITRF', out_frame='ITRF')

    print(p)

    init_data = {
        'a': R_e + 400.0,
        'e': 0.01,
        'inc': 90.0,
        'raan': 10,
        'aop': 10,
        'mu0': 40.0,
        'mjd0': 57125.7729,
        'C_D': 2.3,
        'C_R': 1.0,
        'm': 3.0,
        'A': n.pi*1.0**2,
    }
    t = n.linspace(0,3*3600.0, num=500, dtype=n.float)

    ecefs1 = p.get_orbit(t, **init_data)

    print(p)

    p = PropagatorOrekit(in_frame='EME', out_frame='ITRF')

    ecefs2 = p.get_orbit(t, **init_data)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.plot(ecefs1[0,:], ecefs1[1,:], ecefs1[2,:],".",color="green",label='Initial frame: ITRF')
    ax.plot(ecefs2[0,:], ecefs2[1,:], ecefs2[2,:],".",color="red",label='Initial frame: EME')
    plt.legend()
    plt.show()

'''

#
# Python standard
#
import os
import time

#
# External
#
import numpy as np
import scipy

#
# SORTS++
#
from propagator_base import PropagatorBase
import dpt_tools as dpt

#
# Orekit
#
import orekit
orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir

__prop_file = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])
setup_orekit_curdir(filename = __prop_file + '/data/orekit-data-master.zip')

from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.frames import FramesFactory
from org.orekit.time import TimeScalesFactory, AbsoluteDate
from org.orekit.orbits import KeplerianOrbit, EquinoctialOrbit, CartesianOrbit, PositionAngle
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.python import PythonOrekitStepHandler
import org

from orekit import JArray_double

utc = TimeScalesFactory.getUTC()



def npdt2absdate(dt):
    '''
    Converts a numpy datetime64 value to an orekit AbsoluteDate
    '''

    year, month, day, hour, minutes, seconds, microsecond = dpt.npdt2date(dt)
    return AbsoluteDate(
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minutes),
        float(seconds + microsecond*1e-6),
        utc,
    )


def mjd2absdate(mjd):
    '''
    Converts a Modified Julian Date value to an orekit AbsoluteDate
    '''

    return npdt2absdate(dpt.mjd2npdt(mjd))



def _get_frame(name, frame_tidal_effects = False):
    '''Uses a string to identify which coordinate frame to initialize from Orekit package.

    See `Orekit FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_
    '''
    if name == 'EME':
        return FramesFactory.getEME2000()
    elif name == 'CIRF':
        return FramesFactory.getCIRF(IERSConventions.IERS_2010, not frame_tidal_effects)
    elif name == 'ITRF':
        return FramesFactory.getITRF(IERSConventions.IERS_2010, not frame_tidal_effects)
    elif name == 'TIRF':
        return FramesFactory.getTIRF(IERSConventions.IERS_2010, not frame_tidal_effects)
    elif name == 'ITRFEquinox':
        return FramesFactory.getITRFEquinox(IERSConventions.IERS_2010, not frame_tidal_effects)
    if name == 'TEME':
        return FramesFactory.getTEME()
    else:
        raise Exception('Frame "{}" not recognized'.format(name))


def iter_states(fun):

    def iter_fun(state, mjd, *args, **kwargs):

        if len(state.shape) > 1:
            if state.shape[1] > 1:
                if len(mjd) > 1:
                    assert len(mjd) == state.shape[1], 'State and MJD lengths do not correspond'
                else:
                    raise Exception('Need a epoch for each state')

                state_out = np.empty(state.shape, dtype=state.dtype)
                for ind in range(state.shape[1]):
                    state_tmp = state[:,ind]
                    state_out[:,ind] = fun(state_tmp, mjd[ind], *args, **kwargs)
                return state_out
        
        return fun(state, mjd, *args, **kwargs)

    return iter_fun


@iter_states
def frame_conversion(state, mjd, orekit_in_frame, orekit_out_frame):
    '''Convert Cartesian state between two frames. Currently options are:

    * EME
    * CIRF
    * ITRF
    * TIRF
    * ITRFEquinox
    * TEME

    # TODO: finish docstring
    '''

    state_out = np.empty(state.shape, dtype=state.dtype)

    pos = org.hipparchus.geometry.euclidean.threed.Vector3D(float(state[0]), float(state[1]), float(state[2]))
    vel = org.hipparchus.geometry.euclidean.threed.Vector3D(float(state[3]), float(state[4]), float(state[5]))
    PV_state = PVCoordinates(pos, vel)

    transform = orekit_in_frame.getTransformTo(orekit_out_frame, mjd2absdate(mjd))
    new_PV = transform.transformPVCoordinates(PV_state)


    x_tmp = new_PV.getPosition()
    v_tmp = new_PV.getVelocity()

    state_out[0] = x_tmp.getX()
    state_out[1] = x_tmp.getY()
    state_out[2] = x_tmp.getZ()
    state_out[3] = v_tmp.getX()
    state_out[4] = v_tmp.getY()
    state_out[5] = v_tmp.getZ()

    return state_out





class PropagatorOrekit(PropagatorBase):
    '''Propagator class implementing the Orekit propagator.


    :ivar list solarsystem_perturbers: List of strings of names of objects in the solarsystem that should be used for third body perturbation calculations. All objects listed at `CelestialBodyFactory <https://www.orekit.org/static/apidocs/org/orekit/bodies/CelestialBodyFactory.html>`_ are available.
    :ivar str in_frame: String identifying the input frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :ivar str out_frame: String identifying the output frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :ivar bool drag_force: Should drag force be included in propagation.
    :ivar bool radiation_pressure: Should radiation pressure force be included in propagation.
    :ivar bool frame_tidal_effects: Should coordinate frames include Tidal effects.
    :ivar str integrator: String representing the numerical integrator from the Hipparchus package to use. Any integrator listed at `Hipparchus nonstiff ode <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/package-summary.html>`_ is available.
    :ivar float minStep: Minimum time step allowed in the numerical orbit propagation given in seconds.
    :ivar float maxStep: Maximum time step allowed in the numerical orbit propagation given in seconds.
    :ivar float position_tolerance: Position tolerance in numerical orbit propagation errors given in meters.
    :ivar str earth_gravity: Gravitation model to use for calculating central acceleration force. Currently avalible options are `'HolmesFeatherstone'` and `'Newtonian'`. See `gravity <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/package-summary.html>`_.
    :ivar tuple gravity_order: A tuple of two integers for describing the order of spherical harmonics used in the `HolmesFeatherstoneAttractionModel <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/HolmesFeatherstoneAttractionModel.html>`_ model.
    :ivar str atmosphere: Atmosphere model used to calculate atmospheric drag. Currently available options are `'DTM2000'`. See `atmospheres <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/package-summary.html>`_.
    :ivar str solar_activity: The model used for calculating solar activity and thereby the influx of solar radiation. Used in the atmospheric drag force model. Currently available options are `'Marshall'` for the `MarshallSolarActivityFutureEstimation <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/data/MarshallSolarActivityFutureEstimation.html>`_.
    :ivar str constants_source: Controls which source for Earth constants to use. Currently avalible options are `'WGS84'` and `'JPL-IAU'`. See `constants <https://www.orekit.org/static/apidocs/org/orekit/utils/Constants.html>`_.
    :ivar float mu: Standard gravitational constant for the Earth.  Definition depend on the :class:`propagator_orekit.PropagatorOrekit` constructor parameter :code:`constants_source`
    :ivar float R_earth: Radius of the Earth in m. Definition depend on the :class:`propagator_orekit.PropagatorOrekit` constructor parameter :code:`constants_source`
    :ivar float f_earth: Flattening of the Earth (i.e. :math:`\\frac{a-b}{a}` ). Definition depend on the :class:`propagator_orekit.PropagatorOrekit` constructor parameter :code:`constants_source`.
    :ivar float M_earth: Mass of the Earth in kg. Definition depend on the :class:`propagator_orekit.PropagatorOrekit` constructor parameter :code:`constants_source`
    :ivar org.orekit.frames.Frame inputFrame: The orekit frame instance for the input frame.
    :ivar org.orekit.frames.Frame outputFrame: The orekit frame instance for the output frame.
    :ivar org.orekit.frames.Frame inertialFrame: The orekit frame instance for the inertial frame. If inputFrame is pseudo innertial this is the same as inputFrame.
    :ivar org.orekit.bodies.OneAxisEllipsoid body: The model ellipsoid representing the Earth.
    :ivar dict _forces: Dictionary of forces to include in the numerical integration. Contains instances of children of :class:`org.orekit.forces.AbstractForceModel`.
    :ivar list _tolerances: Contains the absolute and relative tolerances calculated by the `tolerances <https://www.orekit.org/static/apidocs/org/orekit/propagation/numerical/NumericalPropagator.html#tolerances(double,org.orekit.orbits.Orbit,org.orekit.orbits.OrbitType)>`_ function.
    :ivar org.orekit.propagation.numerical.NumericalPropagator propagator: The numerical propagator instance.
    :ivar org.orekit.forces.drag.atmosphere.data.MarshallSolarActivityFutureEstimation.StrengthLevel SolarStrengthLevel: The strength of the solar activity. Options are 'AVRAGE', 'STRONG', 'WEAK'.

    The constructor creates a propagator instance with supplied options.

    :param list solarsystem_perturbers: List of strings of names of objects in the solarsystem that should be used for third body perturbation calculations. All objects listed at `CelestialBodyFactory <https://www.orekit.org/static/apidocs/org/orekit/bodies/CelestialBodyFactory.html>`_ are available.
    :param str in_frame: String identifying the input frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :param str out_frame: String identifying the output frame to be used. All frames listed at `FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_ are available.
    :param bool drag_force: Should drag force be included in propagation.
    :param bool radiation_pressure: Should radiation pressure force be included in propagation.
    :param bool frame_tidal_effects: Should coordinate frames include Tidal effects.
    :param str integrator: String representing the numerical integrator from the Hipparchus package to use. Any integrator listed at `Hipparchus nonstiff ode <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/package-summary.html>`_ is available.
    :param float min_step: Minimum time step allowed in the numerical orbit propagation given in seconds.
    :param float max_step: Maximum time step allowed in the numerical orbit propagation given in seconds.
    :param float position_tolerance: Position tolerance in numerical orbit propagation errors given in meters.
    :param str atmosphere: Atmosphere model used to calculate atmospheric drag. Currently available options are `'DTM2000'`. See `atmospheres <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/package-summary.html>`_.
    :param str solar_activity: The model used for calculating solar activity and thereby the influx of solar radiation. Used in the atmospheric drag force model. Currently available options are `'Marshall'` for the `MarshallSolarActivityFutureEstimation <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/atmosphere/data/MarshallSolarActivityFutureEstimation.html>`_.
    :param str constants_source: Controls which source for Earth constants to use. Currently avalible options are `'WGS84'` and `'JPL-IAU'`. See `constants <https://www.orekit.org/static/apidocs/org/orekit/utils/Constants.html>`_.
    :param str earth_gravity: Gravitation model to use for calculating central acceleration force. Currently avalible options are `'HolmesFeatherstone'` and `'Newtonian'`. See `gravity <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/package-summary.html>`_.
    :param tuple gravity_order: A tuple of two integers for describing the order of spherical harmonics used in the `HolmesFeatherstoneAttractionModel <https://www.orekit.org/static/apidocs/org/orekit/forces/gravity/HolmesFeatherstoneAttractionModel.html>`_ model.
    :param str solar_activity_strength: The strength of the solar activity. Options are 'AVRAGE', 'STRONG', 'WEAK'.
    '''


    class OrekitVariableStep(PythonOrekitStepHandler):
        '''Class for handling the steps.
        '''
        def set_params(self, t, start_date, states_pointer, outputFrame):
            self.t = t
            self.start_date = start_date
            self.states_pointer = states_pointer
            self.outputFrame = outputFrame

        def init(self, s0, t):
            pass

        def handleStep(self, interpolator, isLast):
            state1 = interpolator.getCurrentState()
            state0 = interpolator.getPreviousState()

            t0 = state0.getDate().durationFrom(self.start_date)
            t1 = state1.getDate().durationFrom(self.start_date)

            for ti, t in enumerate(self.t):

                if np.abs(t) >= np.abs(t0) and np.abs(t) <= np.abs(t1):
                    t_date = self.start_date.shiftedBy(float(t))

                    _state = interpolator.getInterpolatedState(t_date)

                    PVCoord = _state.getPVCoordinates(self.outputFrame)

                    x_tmp = PVCoord.getPosition()
                    v_tmp = PVCoord.getVelocity()

                    self.states_pointer[0,ti] = x_tmp.getX()
                    self.states_pointer[1,ti] = x_tmp.getY()
                    self.states_pointer[2,ti] = x_tmp.getZ()
                    self.states_pointer[3,ti] = v_tmp.getX()
                    self.states_pointer[4,ti] = v_tmp.getY()
                    self.states_pointer[5,ti] = v_tmp.getZ()


    def _logger_start(self, name):
        self.__log_tmp[name] = time.time()


    def _logger_record(self, name):
        dt = [time.time() - self.__log_tmp[name]]
        if name in self.__log:
            self.__log[name] += dt
        else:
            self.__log[name] = dt

    def print_logger(self):
        _max_len = 0
        for name in self.__log:
            if len(name) > _max_len:
                _max_len = len(name)
        _head = '{:<' + str(_max_len) + '} | {:<15} | {}'
        _row = '{:<' + str(_max_len) + '} | {:<15} | {:<15.6f}'
        self.logger_func(_head.format('Name', 'Count', 'Mean execution time'))
        for name, vec in self.__log.items():
            self.logger_func(_row.format(name, len(vec), np.mean(vec)))

    def get_logger(self):
        _ret = {}
        for name, vec in self.__log.items():
            _ret[name] = np.mean(vec)
        return _ret



    def __init__(self,
                in_frame='EME',
                out_frame='ITRF',
                frame_tidal_effects=False,
                integrator='DormandPrince853',
                min_step=0.001,
                max_step=120.0,
                position_tolerance=10.0,
                earth_gravity='HolmesFeatherstone',
                gravity_order=(10,10),
                solarsystem_perturbers=['Moon', 'Sun'],
                drag_force=True,
                atmosphere='DTM2000',
                radiation_pressure=True,
                solar_activity='Marshall',
                constants_source='WGS84',
                solar_activity_strength='WEAK',
                logger_func=None,
            ):
        super(PropagatorOrekit, self).__init__()

        self.logger_func = logger_func
        self.__log = {}
        self.__log_tmp = {}
        if self.logger_func is not None:
            self._logger_start('__init__')

        self.solarsystem_perturbers = solarsystem_perturbers
        self.in_frame = in_frame
        self.frame_tidal_effects = frame_tidal_effects
        self.out_frame = out_frame
        self.drag_force = drag_force
        self.gravity_order = gravity_order
        self.atmosphere = atmosphere
        self.radiation_pressure = radiation_pressure
        self.solar_activity = solar_activity
        self.SolarStrengthLevel = getattr(org.orekit.forces.drag.atmosphere.data.MarshallSolarActivityFutureEstimation.StrengthLevel, solar_activity_strength)
        self.integrator = integrator
        self.minStep = min_step
        self.maxStep = max_step
        self.position_tolerance = position_tolerance
        self._tolerances = None
        
        self.constants_source = constants_source
        if constants_source == 'JPL-IAU':
            self.mu = Constants.JPL_SSD_EARTH_GM
            self.R_earth = Constants.IAU_2015_NOMINAL_EARTH_EQUATORIAL_RADIUS
            self.f_earth = (Constants.IAU_2015_NOMINAL_EARTH_EQUATORIAL_RADIUS - Constants.IAU_2015_NOMINAL_EARTH_POLAR_RADIUS)/Constants.IAU_2015_NOMINAL_EARTH_POLAR_RADIUS
        else:
            self.mu = Constants.WGS84_EARTH_MU
            self.R_earth = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
            self.f_earth = Constants.WGS84_EARTH_FLATTENING

        self.M_earth = self.mu/scipy.constants.G

        self.earth_gravity = earth_gravity

        self.__params = None

        self.inputFrame = self._get_frame(self.in_frame)
        self.outputFrame = self._get_frame(self.out_frame)

        if self.inputFrame.isPseudoInertial():
            self.inertialFrame = self.inputFrame
        else:
            self.inertialFrame = FramesFactory.getEME2000()

        self.body = OneAxisEllipsoid(self.R_earth, self.f_earth, self.outputFrame)

        self._forces = {}

        if self.radiation_pressure:
            self._forces['radiation_pressure'] = None
        if self.drag_force:
            self._forces['drag_force'] = None

        if self.earth_gravity == 'HolmesFeatherstone':
            provider = org.orekit.forces.gravity.potential.GravityFieldFactory.getNormalizedProvider(self.gravity_order[0], self.gravity_order[1])
            holmesFeatherstone = org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel(
                FramesFactory.getITRF(IERSConventions.IERS_2010, True),
                provider,
            )
            self._forces['earth_gravity'] = holmesFeatherstone

        elif self.earth_gravity == 'Newtonian':
            Newtonian = org.orekit.forces.gravity.NewtonianAttraction(self.mu)
            self._forces['earth_gravity'] = Newtonian

        else:
            raise Exception('Supplied Earth gravity model "{}" not recognized'.format(self.earth_gravity))

        if self.solarsystem_perturbers is not None:
            for body in self.solarsystem_perturbers:
                body_template = getattr(CelestialBodyFactory, 'get{}'.format(body))
                body_instance = body_template()
                perturbation = org.orekit.forces.gravity.ThirdBodyAttraction(body_instance)

                self._forces['perturbation_{}'.format(body)] = perturbation
        if self.logger_func is not None:
            self._logger_record('__init__')

    def __str__(self):
        if self.logger_func is not None:
            self._logger_start('__str__')

        ret = ''
        ret += 'PropagatorOrekit instance @ {}:'.format(hash(self)) + '\n' + '-'*25 + '\n'
        ret += '{:20s}: '.format('Integrator') + self.integrator + '\n'
        ret += '{:20s}: '.format('Minimum step') + str(self.minStep) + ' s' + '\n'
        ret += '{:20s}: '.format('Maximum step') + str(self.maxStep) + ' s' + '\n'
        ret += '{:20s}: '.format('Position Tolerance') + str(self.position_tolerance) + ' m' + '\n'
        if self._tolerances is not None:
            ret += '{:20s}: '.format('Absolute Tolerance') + str(JArray_double.cast_(self._tolerances[0])) + ' m' + '\n'
            ret += '{:20s}: '.format('Relative Tolerance') + str(JArray_double.cast_(self._tolerances[1])) + ' m' + '\n'
        ret += '\n'
        ret += '{:20s}: '.format('Input frame') + self.in_frame + '\n'
        ret += '{:20s}: '.format('Output frame') + self.out_frame + '\n'
        ret += '{:20s}: '.format('Gravity model') + self.earth_gravity + '\n'
        if self.earth_gravity == 'HolmesFeatherstone':
            ret += '{:20s} - Harmonic expansion order {}'.format('', self.gravity_order) + '\n'
        ret += '{:20s}: '.format('Atmosphere model') + self.atmosphere + '\n'
        ret += '{:20s}: '.format('Solar model') + self.solar_activity + '\n'
        ret += '{:20s}: '.format('Constants') + self.constants_source + '\n'
        ret += 'Included forces:' + '\n'
        for key in self._forces:
            ret += ' - {}'.format(' '.join(key.split('_'))) + '\n'
        ret += 'Third body perturbations:' + '\n'
        for body in self.solarsystem_perturbers:
            ret += ' - {:}'.format(body) + '\n'
        
        if self.logger_func is not None:
            self._logger_record('__str__')

        return ret


    def _get_frame(self, name):
        '''Uses a string to identify which coordinate frame to initialize from Orekit package.

        See `Orekit FramesFactory <https://www.orekit.org/static/apidocs/org/orekit/frames/FramesFactory.html>`_
        '''
        if self.logger_func is not None:
            self._logger_start('_get_frame')

        if name == 'EME':
            return FramesFactory.getEME2000()
        elif name == 'CIRF':
            return FramesFactory.getCIRF(IERSConventions.IERS_2010, not self.frame_tidal_effects)
        elif name == 'ITRF':
            return FramesFactory.getITRF(IERSConventions.IERS_2010, not self.frame_tidal_effects)
        elif name == 'TIRF':
            return FramesFactory.getTIRF(IERSConventions.IERS_2010, not self.frame_tidal_effects)
        elif name == 'ITRFEquinox':
            return FramesFactory.getITRFEquinox(IERSConventions.IERS_2010, not self.frame_tidal_effects)
        if name == 'TEME':
            return FramesFactory.getTEME()
        else:
            raise Exception('Frame "{}" not recognized'.format(name))

        if self.logger_func is not None:
            self._logger_record('_get_frame')


    def _construct_propagator(self, initialOrbit):
        '''
        Get the specified integrator from hipparchus package. List available at: `nonstiff ode <https://www.hipparchus.org/apidocs/org/hipparchus/ode/nonstiff/package-summary.html>`_

        Configure the integrator tolerances using the orbit.
        '''
        if self.logger_func is not None:
            self._logger_start('_construct_propagator')

        self._tolerances = NumericalPropagator.tolerances(
                self.position_tolerance,
                initialOrbit,
                initialOrbit.getType()
            )

        integrator_constructor = getattr(
            org.hipparchus.ode.nonstiff,
            '{}Integrator'.format(self.integrator),
        )

        integrator = integrator_constructor(
            self.minStep,
            self.maxStep, 
            JArray_double.cast_(self._tolerances[0]),
            JArray_double.cast_(self._tolerances[1]),
        )

        propagator = NumericalPropagator(integrator)
        propagator.setOrbitType(initialOrbit.getType())

        self.propagator = propagator

        if self.logger_func is not None:
            self._logger_record('_construct_propagator')



    def _set_forces(self, A, cd, cr):
        '''Using the spacecraft specific parameters, set the drag force and radiation pressure models.
        
        **See:**
            * `drag <https://www.orekit.org/static/apidocs/org/orekit/forces/drag/package-summary.html>`_
            * `radiation <https://www.orekit.org/static/apidocs/org/orekit/forces/radiation/package-summary.html>`_
        '''
        __params = [A, cd, cr]

        if self.logger_func is not None:
            self._logger_start('_set_forces')


        re_calc = True
        if self.__params is None:
            re_calc = True
        else:
            if not np.allclose(np.array(__params,dtype=np.float), self.__params, rtol=1e-3):
                re_calc = True

        if re_calc:        
            self.__params = __params
            if self.drag_force:
                if self.solar_activity == 'Marshall':
                    msafe = org.orekit.forces.drag.atmosphere.data.MarshallSolarActivityFutureEstimation(
                        "(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\p{Digit}\\p{Digit}\\p{Digit}\\p{Digit}F10\\.(?:txt|TXT)",
                        self.SolarStrengthLevel,
                    )
                    manager = org.orekit.data.DataProvidersManager.getInstance()
                    manager.feed(msafe.getSupportedNames(), msafe)

                    if self.atmosphere == 'DTM2000':
                        atmosphere_instance = org.orekit.forces.drag.atmosphere.DTM2000(msafe, CelestialBodyFactory.getSun(), self.body)
                    else:
                        raise Exception('Atmosphere model not recognized')

                    drag_model = org.orekit.forces.drag.DragForce(
                        atmosphere_instance,
                        org.orekit.forces.drag.IsotropicDrag(float(A), float(cd)),
                    )

                    self._forces['drag_force'] = drag_model
                else:
                    raise Exception('Solar activity model not recognized')

            if self.radiation_pressure:
                radiation_pressure_model = org.orekit.forces.radiation.SolarRadiationPressure(
                    CelestialBodyFactory.getSun(),
                    self.body.getEquatorialRadius(),
                    org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient(float(A), float(cr)),
                )

                self._forces['radiation_pressure'] = radiation_pressure_model

            #self.propagator.removeForceModels()

            for force_name, force in self._forces.items():
                self.propagator.addForceModel(force)

        if self.logger_func is not None:
            self._logger_record('_set_forces')

        
    def get_orbit(self,t,a,e,inc,raan,aop,mu0,mjd0, **kwargs):
        '''
        **Implementation:**
    
        Units are in meters and degrees.

        Keyword arguments are:

            * float A: Area in m^2
            * float C_D: Drag coefficient
            * float C_R: Radiation pressure coefficient
            * float m: Mass of object in kg

        *NOTE:*
            * If the eccentricity is below 1e-10 the eccentricity will be set to 1e-10 to prevent Keplerian Jacobian becoming singular.
        

        The implementation first checks if the input frame is Pseudo inertial, if this is true this is used as the propagation frame. If not it is automatically converted to EME (ECI-J2000).

        Since there are forces that are dependent on the space-craft parameters, if these parameter has been changed since the last iteration the numerical integrator is re-initialized at every call of this method. The forces that can be initialized without spacecraft parameters (e.g. Earth gravitational field) are done at propagator construction.

        **Uses:**
            * :func:`propagator_base.PropagatorBase._make_numpy`
            * :func:`propagator_orekit.PropagatorOrekit._construct_propagator`
            * :func:`propagator_orekit.PropagatorOrekit._set_forces`
            * :func:`dpt_tools.kep2cart`
            * :func:`dpt_tools.cart2kep`
            * :func:`dpt_tools.true2mean`
            * :func:`dpt_tools.mean2true`
            
        
        See :func:`propagator_base.PropagatorBase.get_orbit`.
        '''


        if self.logger_func is not None:
            self._logger_start('get_orbit')


        if self.radiation_pressure:
            if 'C_R' not in kwargs:
                raise Exception('Radiation pressure force enabled but no coefficient "C_R" given')
        else:
            kwargs['C_R'] = 1.0

        if self.drag_force:
            if 'C_D' not in kwargs:
                raise Exception('Drag force enabled but no drag coefficient "C_D" given')
        else:
            kwargs['C_D'] = 1.0

        t = self._make_numpy(t)

        initialDate = mjd2absdate(mjd0)

        if not self.inputFrame.isPseudoInertial():
            orb = np.array([a, e, inc, aop, raan, dpt.mean2true(mu0, e, radians=False)], dtype=np.float)
            cart = dpt.kep2cart(
                orb,
                m=kwargs['m'],
                M_cent=self.M_earth,
                radians=False,
            )

            pos = org.hipparchus.geometry.euclidean.threed.Vector3D(float(cart[0]), float(cart[1]), float(cart[2]))
            vel = org.hipparchus.geometry.euclidean.threed.Vector3D(float(cart[3]), float(cart[4]), float(cart[5]))
            PV_state = PVCoordinates(pos, vel)

            transform = self.inputFrame.getTransformTo(self.inertialFrame, initialDate)
            new_PV = transform.transformPVCoordinates(PV_state)

            initialOrbit = CartesianOrbit(
                new_PV,
                self.inertialFrame,
                initialDate,
                self.mu + float(scipy.constants.G*kwargs['m']),
            )
        else:

            #this is to prevent Keplerian Jacobian to become singular
            use_equinoctial = False
            if e < dpt.e_lim:
                use_equinoctial = True
            if inc < dpt.i_lim:
                use_equinoctial = True
            
            if use_equinoctial:
                ex = e*np.cos(np.radians(aop) + np.radians(raan))
                ey = e*np.sin(np.radians(aop) + np.radians(raan))
                hx = np.tan(np.radians(inc)*0.5)*np.cos(np.radians(raan))
                hy = np.tan(np.radians(inc)*0.5)*np.sin(np.radians(raan))
                lv = np.radians(mu0) + np.radians(aop) + np.radians(raan)

                initialOrbit = EquinoctialOrbit(
                    float(a),
                    float(ex),
                    float(ey),
                    float(hx),
                    float(hy),
                    float(lv),
                    PositionAngle.MEAN,
                    self.inertialFrame,
                    initialDate,
                    self.mu + float(scipy.constants.G*kwargs['m']),
                )
            else:
                initialOrbit = KeplerianOrbit(
                    float(a),
                    float(e),
                    float(np.radians(inc)),
                    float(np.radians(aop)),
                    float(np.radians(raan)),
                    float(np.radians(mu0)),
                    PositionAngle.MEAN,
                    self.inertialFrame,
                    initialDate,
                    self.mu + float(scipy.constants.G*kwargs['m']),
                )


        self._construct_propagator(initialOrbit)
        self._set_forces(kwargs['A'], kwargs['C_D'], kwargs['C_R'])

        initialState = SpacecraftState(initialOrbit)

        self.propagator.setInitialState(initialState)

        tb_inds = t < 0.0
        t_back = t[tb_inds]

        tf_indst = t >= 0.0
        t_forward = t[tf_indst]

        if len(t_forward) == 1:
            if np.any(t_forward == 0.0):
                t_back = t
                t_forward = []
                tb_inds = t <= 0

        state = np.empty((6, len(t)), dtype=np.float)
        step_handler = PropagatorOrekit.OrekitVariableStep()

        if self.logger_func is not None:
            self._logger_start('propagation')

        if len(t_back) > 0:
            _t = t_back
            _t_order = np.argsort(np.abs(_t))
            _t_res = np.argsort(_t_order)
            _t = _t[_t_order]
            _state = np.empty((6, len(_t)), dtype=np.float) 
            step_handler.set_params(_t, initialDate, _state, self.outputFrame)

            self.propagator.setMasterMode(step_handler)

            self.propagator.propagate(initialDate.shiftedBy(float(_t[-1])))
            
            #now _state is full and in the order of _t
            state[:, tb_inds] = _state[:, _t_res]

        if len(t_forward) > 0:
            _t = t_forward
            _t_order = np.argsort(np.abs(_t))
            _t_res = np.argsort(_t_order)
            _t = _t[_t_order]
            _state = np.empty((6, len(_t)), dtype=np.float) 
            step_handler.set_params(_t, initialDate, _state, self.outputFrame)

            self.propagator.setMasterMode(step_handler)

            self.propagator.propagate(initialDate.shiftedBy(float(_t[-1])))
            
            #now _state is full and in the order of _t
            state[:, tf_indst] = _state[:, _t_res]

        if self.logger_func is not None:
            self._logger_record('propagation')

        if self.logger_func is not None:
            self._logger_record('get_orbit')

        return state

    def get_orbit_cart(self, t, x, y, z, vx, vy, vz, mjd0, **kwargs):
        '''
        **Implementation:**

        Converts Cartesian vector to Kepler elements and calls :func:`propagator_orekit.PropagatorOrekit.get_orbit`.

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
            M_cent=self.M_earth,
            radians=False,
        )

        a = orb[0]
        e = orb[1]
        inc = orb[2]
        aop = orb[3]
        raan = orb[4]
        mu0 = dpt.true2mean(orb[5], e, radians=False)

        return self.get_orbit(t, a, e, inc, raan, aop, mu0, mjd0, **kwargs)

