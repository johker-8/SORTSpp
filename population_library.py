'''Library of population instances.

'''
import os
import time
import copy

os.environ['TZ'] = 'GMT'
time.tzset()

import numpy as np
import scipy.constants as consts
import h5py

from population import Population
import population_filter as pf
import TLE_tools as tle
import dpt_tools as dpt
from mpi4py import MPI

import debris

comm = MPI.COMM_WORLD

from sorts_config import p as default_propagator

try:
    import propagator_sgp4
except ImportError:
    propagator_sgp4 = None

try:
    import propagator_orekit
except ImportError:
    propagator_orekit = None

def master_catalog(
        input_file = "./master/celn_20090501_00.sim",
        mjd0 = 54952.0,
        sort=True,
        propagator = default_propagator,
        propagator_options = {},
    ):
    '''Return the master catalog specified in the input file as a population instance. The catalog only contains the master sampling objects and not an actual realization of the population using the factor.

    The format of the input master files is:

        0. ID
        1. Factor
        2. Mass [kg]
        3. Diameter [m]
        4. m/A [kg/m2]
        5. a [km]
        6. e
        7. i [deg]
        8. RAAN [deg]
        9. AoP [deg]
        10. M [deg]


    :param str input_file: Path to the input MASTER file.
    :param bool sort: If :code:`True` sort according to diameters in descending order.
    :param float mjd0: The epoch of the catalog file in Modified Julian Days.
    :param PropagatorBase propagator: Propagator class pointer used for :class:`space_object.SpaceObject`.
    :param dict propagator_options: Propagator initialization keyword arguments.
    

    :return: Master catalog
    :rtype: population.Population
    '''
    master_raw = np.genfromtxt(input_file);
    i = [0,5,6,7,8,9,10]

    master = Population(
        name='MASTER-2009',
        extra_columns = ['A', 'm', 'd', 'C_D', 'C_R', 'Factor'],
        space_object_uses = [True, True, True, True, True, False],
        propagator = propagator,
        propagator_options = propagator_options,
    )

    master.allocate(master_raw.shape[0])

    master[:,:7] = master_raw[:, i]
    master.objs['mjd0'] = mjd0
    master.objs['A'] = np.divide(master_raw[:, 2], master_raw[:, 4])
    master.objs['m'] = master_raw[:, 2]
    master.objs['d'] = master_raw[:, 3]
    master.objs['C_D'] = 2.3
    master.objs['C_R'] = 1.0
    master.objs['Factor'] = master_raw[:, 1]

    diams = master_raw[:, 3]

    if sort:
        idxs = np.argsort(diams)[::-1]
    else:
        idxs = np.arange(len(diams))

    master.objs = master.objs[idxs]
    
    return master

def master_catalog_factor(
            input_file = "./master/celn_20090501_00.sim",
            mjd0 = 54952.0,
            master_base=None,
            treshhold = 0.01,
            seed=None,
            propagator = default_propagator,
            propagator_options = {},
        ):
    '''Returns a random realization of the master population specified by the input file/population. In other words, each sampling object in the catalog is sampled a "factor" number of times with random mean anomalies to create the population.

    :param str input_file: Path to the input MASTER file. Is not used if :code:`master_base` is given.
    :param float mjd0: The epoch of the catalog file in Modified Julian Days. Is not used if :code:`master_base` is given.
    :param population.Population master_base: A master catalog consisting only of sampling objects. This catalog will be modified and the pointer to it returned.
    :param bool sort: If :code:`True` sort according to diameters in ascending order.
    :param float treshhold: Diameter limit in meters below witch sampling objects are not included. Can be :code:`None` to skip filtering.
    :param int seed: Random number generator seed given to :code:`numpy.random.seed` to allow for consisted generation of a random realization of the population. If seed is :code:`None` a random seed from high-entropy data is used.
    :param PropagatorBase propagator: Propagator class pointer used for :class:`space_object.SpaceObject`. Is not used if :code:`master_base` is given.
    :param dict propagator_options: Propagator initialization keyword arguments. Is not used if :code:`master_base` is given.
    

    :return: Master population
    :rtype: population.Population
    '''
    np.random.seed(seed=seed)

    if master_base is None:
        master_base = master_catalog(
            input_file = input_file,
            mjd0 = mjd0,
            sort=True,
            propagator = propagator,
            propagator_options = propagator_options,
        )

    master_base.name += '-Factor'
    
    if treshhold is not None:
        master_base.filter('d', lambda d: d >= treshhold)

    full_objs = np.zeros((int(np.sum(np.round(master_base['Factor']))),master_base.shape[1]), dtype=np.float)
    
    max_oid = np.max(master_base['oid'])
    max_factor = np.floor(np.log10(np.max(master_base['Factor'])))
    oid_extend = 10**(int(np.log10(max_oid))+max_factor+1.0)
    oid_mag = 10**(int(np.log10(max_oid)))

    i=0
    for row in master_base.objs:
        f_int = int(np.round(row[13]))
        if f_int >= 1:
            ip = i+f_int
            for coli, head in enumerate(master_base.header):
                full_objs[i:ip,coli] = row[head]

            id_str = ['1{:03d}{:09d}'.format(int(ind+1), int(row[0])) for ind in range(f_int)]
            id_str = [ int(x) for x in id_str]

            full_objs[i:ip,0] = np.array(id_str, dtype=np.float)
            full_objs[i:ip,6] = np.random.rand(f_int)*np.pi*2.0
            i=ip

    master_base.allocate(full_objs.shape[0])

    master_base[:,:] = full_objs


    return master_base

def filtered_master_catalog_factor(
            radar,
            input_file="./master/celn_20090501_00.sim",
            detectability_file=None,
            mjd0=54952.0,
            treshhold = 0.01,
            min_inc=50,
            seed=65487945,
            prop_time=24.0,
            propagator = default_propagator,
            propagator_options = {},
        ):
    '''Returns a random realization of the master population specified by the input file/population but filtered according detectability from a :class:`radar_config.RadarSystem`.

    Filter results are saved in the same folder as the :code:`input_file` variable specifies the Master catalog file location.
    
    :param RadarSystem radar: The radar configuration used for the detectability filtering.
    :param str input_file: Path to the input MASTER file. Is not used if :code:`master_base` is given.
    :param str detectability_file: Path to the output-definition file so that a cached file can be used to load population instead of re-calculating every time.
    :param bool sort: If :code:`True` sort according to diameters in ascending order.
    :param float treshhold: Diameter limit in meters below witch sampling objects are not included. Can be :code:`None` to skip filtering.
    :param float min_inc: Inclination limit in degrees below witch sampling objects are not included. Can be :code:`None` to skip filtering.
    :param int seed: Random number generator seed given to :code:`numpy.random.seed` to allow for consisted generation of a random realization of the population. If seed is :code:`None` a random seed from high-entropy data is used.
    :param float prop_time: Propagation time used to check if object is detectable.
    :param PropagatorBase propagator: Propagator class pointer used for :class:`space_object.SpaceObject`.
    :param dict propagator_options: Propagator initialization keyword arguments.

    :return: Filtered master population
    :rtype: population.Population
    '''
    mc = master_catalog(
            input_file = input_file,
            mjd0 = mjd0,
            sort=True,
            propagator = propagator,
            propagator_options = propagator_options,
        )
    if detectability_file is None:
        output_file = input_file[:-4]\
            + '_' + radar.name.replace(' ','_')\
            + '_' + propagator.__name__\
            + '_' + str(int(np.round(radar.min_SNRdb))) + 'SNRdB'\
            + '.h5'
    else:
        output_file = detectability_file
    mc.name = 'Detectable ' + mc.name
    
    if treshhold is not None:
        mc.filter('d', lambda x: x >= treshhold)
    if min_inc is not None:
        mc.filter('i', lambda x: x >= min_inc)

    #FOR DEBUG
    #mc._objs = mc._objs[:20,]

    if not os.path.isfile(output_file):
        pf.filter_objects(radar, mc, ofname=output_file, prop_time=prop_time)

    h=h5py.File(output_file,"r")
    detectable=np.copy(h["detectable"].value)
    h.close()

    mc.objs = mc.objs[detectable]
    mc = master_catalog_factor(master_base = mc, treshhold = None, seed = seed)
    return mc

def _find_range_for_snr(radar, d, start_r ,SNR):

    def snr_func(rang):
        snrs = []
        for txi, tx in enumerate(radar._tx):
            for rxi, rx in enumerate(radar._rx):
                snr_tmp = debris.hard_target_enr(
                    tx.beam.I_0,
                    tx.beam.I_0,
                    rx.wavelength,
                    tx.tx_power,
                    rang,
                    rang,
                    diameter_m = d,
                    bandwidth = tx.coh_int_bandwidth,
                    rx_noise_temp = rx.rx_noise,
                )
                snrs.append(snr_tmp)
        return np.max(np.array(snrs))

    max_snr = snr_func(start_r)
    rn = start_r
    delta_rn_base = 10000e3
    delta_rn = delta_rn_base
    while max_snr < SNR:
        rn -= delta_rn
        max_snr = snr_func(rn)
        
        delta_rn = delta_rn_base*(1.0 - max_snr/SNR + 0.0001)

    return rn, max_snr
        


def NESCv9_mini_moons(albedo, propagate_SNR = None, radar = None, truncate = None):
    from propagator_rebound import PropagatorRebound

    _opts = dict(
        in_frame = 'HeliocentricECLIPJ2000',
        out_frame = 'ITRF',
    )

    #fname = './data/NESCv9reintv1.TCO.withoutH.kep.des'
    fname = './data/NESCv9reintv1.TCO.withH.kep.des'

    data = np.genfromtxt(fname)

    if truncate is not None:
        _trunc = np.full((data.shape[0],), True, dtype=np.bool)
        _trunc[truncate] = False

        data = data[_trunc, :]
        print('Data truncated to: {}'.format(data.shape[0]))

    AU = 149597871.0 #km

    max_check = 365.0*3600.0*24.0

    '''
    1. synthetic orbit designation
    2. orbital element type (KEP for heliocentric Keplerian)
    3. semimajor axis (au)
    4. eccentricity
    5. inclination (deg)
    6. longitude of ascending node (deg)
    7. argument of perihelion (deg)
    8. mean anomaly (deg)
    9. H-magnitude (filler value)
    10. epoch for which the input orbit is valid (modified Julian date)
    11. index (1)
    12. number of parameters (6)
    13. MOID (filler value)
    14. Code with which created (OPENORB)
    '''

    pop = Population(
        name='NESCv9',
        extra_columns = ['d'],
        space_object_uses = [True],
        propagator = PropagatorRebound,
        propagator_options = _opts,
    )

    pop.allocate(data.shape[0])

    pop['oid'] = np.arange(len(pop))
    pop['a'] = data[:,2]*AU
    pop['e'] = data[:,3]
    pop['i'] = data[:,4]
    pop['raan'] = data[:,5]
    pop['aop'] = data[:,6]
    pop['mu0'] = data[:,7]
    pop['mjd0'] = data[:,9]
    pop['d'] = 10.0**(3.1236 - 0.5*np.log10(albedo) - 0.2*data[:,8])

    if propagate_SNR is not None:
        my_iter = list(range(comm.rank, len(pop), comm.size))

        ranges = np.empty((len(pop),), dtype=np.float64)
        for ind, _d in enumerate(pop['d']):
            if ind in my_iter:
                ranges[ind], _snr = _find_range_for_snr(radar, _d, AU*1e3 ,propagate_SNR)
                print('Object {} found range {:.3f} LD @ {:.2f} SNR'.format(ind, ranges[ind]/384400e3, _snr))

        t_at_dist = np.empty((len(pop),), dtype=np.float64)
        
        new_elements = np.empty((6,len(pop)), dtype=np.float64)
        prop_ok = np.full((len(pop),), True, dtype=np.bool)

        obj_gen = pop.object_generator()

        t_v = np.arange(0.0, max_check, 3600.0)
        for ind, obj in enumerate(obj_gen):
            if ind in my_iter:
                print('Checking object {}'.format(ind))
                states = obj.get_orbit(t_v)
                dists = np.linalg.norm(states, axis=0)
                _check = dists < ranges[ind]
                if np.any(_check):
                    t_at_dist[ind] = t_v[np.argmax(_check)]
                    print('|- Passed @ {:.2f} days'.format(t_at_dist[ind]/(24.0*3600.0)))
                else:
                    prop_ok[ind] = False
        
        if 'out_frame' in pop.propagator_options:
            out_f = pop.propagator_options['out_frame']
        else:
            out_f = 'ITRF'
        pop.propagator_options['out_frame'] = _opts['in_frame']

        obj_gen = pop.object_generator()

        for ind, obj in enumerate(obj_gen):
            if ind in my_iter and prop_ok[ind]:
                print('PID{}: propagating obj {} {:.2f} h'.format(comm.rank, ind, t_at_dist[ind]/3600.0))

                try:
                    state = obj.get_state(t_at_dist[ind])
                except:
                    prop_ok[ind] = False
                    continue

                kep = dpt.cart2kep(state, m=obj.m, M_cent=obj.M_cent, radians=False)

                if np.any(np.isnan(kep)):
                    prop_ok[ind] = False
                    continue

                new_elements[0,ind] = kep[0]*1e-3
                new_elements[1,ind] = kep[1]
                new_elements[2,ind] = kep[2]
                new_elements[3,ind] = kep[4]
                new_elements[4,ind] = kep[3]
                new_elements[5,ind] = dpt.true2mean(kep[5], kep[1], radians=False)

        for pid in range(comm.size):
            for ind in range(pid, len(pop), comm.size):
                new_elements[:,ind] = comm.bcast(new_elements[:,ind], root=pid)
                prop_ok[ind] = comm.bcast(prop_ok[ind], root=pid)
                t_at_dist[ind] = comm.bcast(t_at_dist[ind], root=pid)

        pop.objs['a'] = new_elements[0,:]
        pop.objs['e'] = new_elements[1,:]
        pop.objs['i'] = new_elements[2,:]
        pop.objs['raan'] = new_elements[3,:]
        pop.objs['aop'] = new_elements[4,:]
        pop.objs['mu0'] = new_elements[5,:]
        pop.objs['mjd0'] += t_at_dist/(3600.0*24.0)

        pop.objs = pop.objs[prop_ok]

        pop.propagator_options['out_frame'] = out_f


    return pop


def tle_snapshot(tle_file, sgp4_propagation=True, propagator = default_propagator, propagator_options = {}):
    '''Reads a TLE-snapshot file and converts the TLE's to orbits in a TEME frame and creates a population file. The BSTAR parameter is saved in column BSTAR (or :code:`_objs[:,12`). A snapshot generally contains several TLE's for the same object thus will this population also contain duplicate objects.
    
    *Numerical propagator assumptions:*
    To propagate with a numerical propagator one needs to make assumptions.
       * Density is :math:`5\cdot 10^3 \;\frac{kg}{m^3}`.
       * Object is a sphere
       * Drag coefficient is 2.3.


    :param str/list tle_file: Path to the input TLE snapshot file. Or the TLE-set can be given directly as a list of two lines that can be unpacked in a loop, e.g. :code:`[(tle1_l1, tle1_l2), (tle2_l1, tle2_l2)]`.
    :param bool sgp4_propagation: If :code:`False` then the population is specifically constructed to be propagated with :class:`propagator_orekit.PropagatorOrekit` and assumptions are made on mass, density and shape of objects. Otherwise the :class:`space_object.SpaceObject` is configured to use SGP4 propagation.

    :return: TLE snapshot as a population with numerical propagator
    :rtype: population.Population
    '''
    if isinstance(tle_file, str):
        tle_raw = [line.rstrip('\n') for line in open(tle_file)]
        if len(tle_raw) % 2 != 0:
            raise Exception('Not even number of lines [not TLE compatible]')

        TLEs = zip(tle_raw[0::2], tle_raw[1::2])
        pop_name = tle_file.split('/')[-1]
    else:
        TLEs = tle_file
        pop_name = 'TLE database'

    M_earth = propagator_sgp4.SGP4.GM*1e9/consts.G

    if sgp4_propagation:
        pop = Population(
            name=pop_name,
            extra_columns = ['A', 'm', 'd', 'C_D', 'C_R'] + ['line1', 'line2'],
            dtypes = ['float64']*5 + ['U70', 'U70'],
            space_object_uses = [True, True, True, True, True] + [True, True],
            propagator = propagator_sgp4.PropagatorTLE,
            propagator_options = {
                'out_frame': 'ITRF',
                'polar_motion': False,
            },
        )
    else:
        pop = Population(
            name=pop_name,
            extra_columns = ['A', 'm', 'd', 'C_D', 'C_R'],
            space_object_uses = [True, True, True, True, True],
            propagator = propagator,
            propagator_options = propagator_options,
        )

    pop.allocate(len(TLEs))

    for line_id, lines in enumerate(TLEs):
        line1, line2 = lines

        sat_id = tle.tle_id(line1)
        jd0 = tle.tle_jd(line1)
        mjd0 = dpt.jd_to_mjd(jd0)

        state_TEME, epoch = tle.TLE_to_TEME(line1,line2)
        kep = dpt.cart2kep(state_TEME, m=0.0, M_cent=M_earth, radians=False)
        pop.objs[line_id][1] = kep[0]*1e-3
        pop.objs[line_id][2] = kep[1]
        pop.objs[line_id][3] = kep[2]
        pop.objs[line_id][4] = kep[4]
        pop.objs[line_id][5] = kep[3]
        pop.objs[line_id][6] = dpt.true2mean(kep[5], kep[1], radians=False)
        
        pop.objs[line_id][0] = float(sat_id)
        pop.objs[line_id][7] = mjd0

    if sgp4_propagation:
        for line_id, lines in enumerate(TLEs):
            line1, line2 = lines
            pop.objs[line_id][13] = line1[:-1]
            pop.objs[line_id][14] = line2[:-1]
    
    for line_id, lines in enumerate(TLEs):
        line1, line2 = lines

        bstar = tle.tle_bstar(line1)/(propagator_sgp4.SGP4.R_EARTH*1000.0)
        B = bstar*2.0/propagator_sgp4.SGP4.RHO0
        if B < 1e-9:
            rho = 500.0
            C_D = 0.0
            r = 0.1
            A = np.pi*r**2
            m = rho*4.0/3.0*np.pi*r**3
        else:
            C_D = 2.3
            rho = 5.0
            r = (3.0*C_D)/(B*rho)
            A = np.pi*r**2
            m = rho*4.0/3.0*np.pi*r**3

        pop.objs[line_id][8] = A
        pop.objs[line_id][9] = m
        pop.objs[line_id][10] = r*2.0
        pop.objs[line_id][11] = C_D
        pop.objs[line_id][12] = 1.0

    return pop


def propagate_population(population, mjd):

    obj_gen = population.object_generator()

    new_elements = np.empty((6,len(population)), dtype=np.float64)
    prop_ok = np.full((len(population),), True, dtype=np.bool)

    my_iter = list(range(comm.rank, len(population), comm.size))

    for ind, obj in enumerate(obj_gen):
        if ind in my_iter:
            tt = (mjd - obj.mjd0)*3600.0*24.0
            print('PID{}: propagating obj {} {:.2f} h'.format(comm.rank, ind, tt/3600.0))

            try:
                state = obj.get_state(tt)
            except:
                prop_ok[ind] = False
                continue

            kep = dpt.cart2kep(state, m=obj.m, M_cent=obj.M_cent, radians=False)

            if np.any(np.isnan(kep))\
             or kep[1] > 1.0:
                prop_ok[ind] = False
                continue

            new_elements[0,ind] = kep[0]*1e-3
            new_elements[1,ind] = kep[1]
            new_elements[2,ind] = kep[2]
            new_elements[3,ind] = kep[4]
            new_elements[4,ind] = kep[3]
            new_elements[5,ind] = dpt.true2mean(kep[5], kep[1], radians=False)

    for pid in range(comm.size):
        for ind in range(pid, len(population), comm.size):
            new_elements[:,ind] = comm.bcast(new_elements[:,ind], root=pid)
            prop_ok[ind] = comm.bcast(prop_ok[ind], root=pid)

    population.objs['a'] = new_elements[0,:]
    population.objs['e'] = new_elements[1,:]
    population.objs['i'] = new_elements[2,:]
    population.objs['raan'] = new_elements[3,:]
    population.objs['aop'] = new_elements[4,:]
    population.objs['mu0'] = new_elements[5,:]
    population.objs['mjd0'] = mjd

    population.objs = population.objs[prop_ok]

    return population

def _rand_sph(num):

    vecs = []
    while len(vecs) < num:
        on_sph = False
        while not on_sph:
            xi = np.random.rand(4)*2.0 - 1
            xin = xi.dot(xi)
            if xin <= 1.0:
                on_sph = True

        x = 2.0*(xi[1]*xi[3] + xi[0]*xi[2])/xin
        y = 2.0*(xi[2]*xi[3] - xi[0]*xi[1])/xin
        z = (xi[0]**2 + xi[3]**2 - xi[1]**2 - xi[2]**2)/xin

        vecs.append([x,y,z])

    return np.array(vecs, dtype=np.float64)


def simulate_Microsat_R_debris(num, max_dv, radii_range, mass_range, C_D_range, seed, propagator, propagator_options, mjd):

    tle_raw = '''1 43947U 19006A   19069.62495353  .00222140  22344-4  39735-3 0  9995
    2 43947  96.6263 343.1609 0026772 226.5664 224.4731 16.01032328  7176
    1 43947U 19006A   19069.68634861  .00217235  21442-4  38851-3 0  9995
    2 43947  96.6282 343.2231 0026267 227.5469 217.1086 16.01062873  7183
    1 43947U 19006A   19069.81093322  .00066734  37128-5  12017-3 0  9992
    2 43947  96.6282 343.3467 0026246 227.0726 215.1493 16.01048534  7204
    1 43947U 19006A   19070.54563657  .00111882  69857-5  20248-3 0  9991
    2 43947  96.6271 344.0664 0025783 228.7058 125.4384 16.00925959  7318
    1 43947U 19006A   19070.80923755  .00013653  19858-5  25403-4 0  9991
    2 43947  96.6268 344.3320 0024927 225.1848 207.3499 16.01142181  7364
    1 43947U 19006A   19071.12831853  .00154965  11740-4  27719-3 0  9998
    2 43947  96.6279 344.6467 0025572 226.6832 243.5148 16.01113233  7412
    1 43947U 19006A   19071.80921646  .00118874  76545-5  21379-3 0  9993
    2 43947  96.6279 345.3224 0025491 224.0468 208.0149 16.01043615  7523
    1 43947U 19006A   19072.44153689  .00037015  24665-5  66863-4 0  9999
    2 43947  96.6279 345.9498 0025442 221.6016 252.7257 16.01159223  7620
    1 43947U 19006A   19072.74405995  .00022228  21113-5  40609-4 0  9997
    2 43947  96.6279 346.2500 0025432 220.3955 196.4626 16.01147542  7675
    1 43947U 19006A   19073.12825718  .00008242  19370-5  15873-4 0  9996
    2 43947  96.6304 346.6133 0023002 223.0617 246.6563 16.01114545  7738
    1 43947U 19006A   19073.44154376  .00035052  24102-5  63784-4 0  9998
    2 43947  96.6300 346.9244 0022588 225.2511 249.1037 16.01188227  7782
    1 43947U 19006A   19073.80920446 -.00023017  21041-5 -40417-4 0  9994
    2 43947  96.6337 347.2908 0022039 223.4278 208.5467 16.01076766  7841
    1 43947U 19006A   19074.74408622  .00119327  76982-5  21614-3 0  9994
    2 43947  96.6289 348.2321 0022856 222.6623 194.1149 16.01049775  7992
    1 43947U 19006A   19075.01672454  .00167822  13470-4  30180-3 0  9994
    2 43947  96.6290 348.5015 0022778 221.4659 325.7020 16.01154108  8039
    1 43947U 19006A   19075.54702546  .00026769  22010-5  48973-4 0  9993
    2 43947  96.6290 349.0278 0022754 219.3583 142.5100 16.01177853  8128
    1 43947U 19006A   19075.74902949  .00160298  12418-4  29196-3 0  9995
    2 43947  96.6301 349.2308 0021693 223.2409 221.9908 16.00986874  8150
    1 43947U 19006A   19075.80608134  .00151204  11245-4  27527-3 0  9991
    2 43947  96.6283 349.2862 0021642 222.7701 191.0541 16.01000711  8166
    1 43947U 19006A   19075.80608134  .00152613  11422-4  27780-3 0  9994
    2 43947  96.6283 349.2862 0021642 222.7701 191.0553 16.01002279  8160
    1 43947U 19006A   19076.04950662  .00212767  20643-4  38419-3 0  9990
    2 43947  96.6283 349.5277 0021615 221.8443 154.0766 16.01125308  8205
    1 43947U 19006A   19076.38314227  .00228397  23614-4  40852-3 0  9999
    2 43947  96.6283 349.8588 0021611 220.6464 277.1313 16.01285801  8254
    1 43947U 19006A   19076.56965602  .00375665  62286-4  68387-3 0  9990
    2 43947  96.6317 350.0396 0020897 222.9971 269.0493 16.01007118  8281
    1 43947U 19006A   19076.62489356  .00301460  40170-4  54811-3 0  9991
    2 43947  96.6317 350.0945 0020879 222.7479 227.4454 16.01029701  8290
    1 43947U 19006A   19076.74888822  .00206456  19517-4  37455-3 0  9990
    2 43947  96.6317 350.2206 0020779 222.3519 222.0257 16.01081824  8311
    1 43947U 19006A   19076.80334391  .00189842  16753-4  34417-3 0  9997
    2 43947  96.6317 350.2747 0020760 222.1378 175.8977 16.01098177  8326
    1 43947U 19006A   19077.25548485  .00042586  26475-5  77957-4 0  9991
    2 43947  96.6337 350.7139 0018680 228.2848 254.2224 16.01199217  8396
    1 43947U 19006A   19077.49923692  .00048049  28483-5  87602-4 0  9993
    2 43947  96.6337 350.9559 0018675 227.3154 219.3457 16.01241589  8433
    1 43947U 19006A   19077.73867148  .00176479  14671-4  32436-3 0  9999
    2 43947  96.6324 351.1992 0019599 225.2817 160.3011 16.00913438  8475
    1 43947U 19006A   19078.62402633  .00193781  17362-4  35487-3 0  9996
    2 43947  96.6337 352.0814 0019337 224.7610 220.1468 16.00981700  8611
    1 43947U 19006A   19079.44446882  .00056887  32197-5  10424-3 0  9994
    2 43947  96.6318 352.8859 0017932 222.6020 267.9762 16.01149860  8742
    1 43947U 19006A   19079.56114259  .00331197  48376-4  60603-3 0  9996
    2 43947  96.6357 353.0133 0018767 223.2624 219.3325 16.01004143  8769
    1 43947U 19006A   19079.87121502  .00017670  20377-5  33169-4 0  9996
    2 43947  96.6348 353.3097 0018626 221.5456 207.0221 16.01103300  8819
    1 43947U 19006A   19080.12837572  .00018161  20446-5  34105-4 0  9991
    2 43947  96.6342 353.5628 0017856 223.7454 246.0274 16.01109643  8852
    1 43947U 19006A   19080.25184005  .00021666  21015-5  40443-4 0  9990
    2 43947  96.6342 353.6854 0017855 223.2528 237.6932 16.01123784  8871
    1 43947U 19006A   19080.56838169  .00007182  19289-5  14116-4 0  9996
    2 43947  96.6323 353.9969 0018497 223.3308 260.9364 16.01125965  8927
    1 43947U 19006A   19080.80934521 -.00010028  19397-5 -17132-4 0  9998
    2 43947  96.6323 354.2362 0018496 222.3649 209.7823 16.01086764  8962
    1 43947U 19006A   19081.01817130 -.00000602  19026-5  00000+0 0  9994
    2 43947  96.6305 354.4520 0018602 217.7557 337.0831 16.00999357  8992
    1 43947U 19006A   19081.19970958  .00192780  17239-4  35047-3 0  9995
    2 43947  96.6332 354.6387 0017808 222.1497 298.3751 16.01166044  9020
    1 43947U 19006A   19081.44288380  .00132357  90623-5  24030-3 0  9997
    2 43947  96.6332 354.8802 0017799 221.2143 260.0465 16.01211967  9068
    1 43947U 19006A   19081.68209472  .00185493  16056-4  34001-3 0  9993
    2 43947  96.6354 355.1161 0017576 220.8387 198.1644 16.01033820  9107
    1 43947U 19006A   19081.80977756  .00126608  84347-5  23219-3 0  9995
    2 43947  96.6356 355.2444 0017575 222.5408 211.8678 16.01049036  9124
    1 43947U 19006A   19082.19785193  .00009851  19489-5  19026-4 0  9997
    2 43947  96.6356 355.6299 0017571 221.0666 288.6619 16.01097025  9186
    1 43947U 19006A   19082.25710510  .00009887  19472-5  19026-4 0  9996
    2 43947  96.6318 355.6880 0017118 220.4674 270.5748 16.01171238  9192
    1 43947U 19006A   19082.38407701  .00138221  97199-5  25077-3 0  9994
    2 43947  96.6325 355.8124 0017241 219.6702 282.7837 16.01240682  9219
    1 43947U 19006A   19082.68646591  .00149753  11067-4  27555-3 0  9993
    2 43947  96.6363 356.1139 0016876 222.4115 221.6813 16.01004284  9261
    1 43947U 19006A   19082.81114447  .00088395  50732-5  16304-3 0  9995
    2 43947  96.6363 356.2377 0016873 221.9425 220.2429 16.01009980  9289
    1 43947U 19006A   19083.00642146  .00143972  10375-4  26369-3 0  9992
    2 43947  96.6358 356.4314 0016878 221.0757 265.8708 16.01087252  9317
    1 43947U 19006A   19083.49288397 -.03265362  32488-2 -61588-2 0  9990
    2 43947  96.6406 356.9162 0017256 219.6745 189.3161 16.00630469  9392
    1 43947U 19006A   19083.74911125  .00084825  48174-5  15749-3 0  9999
    2 43947  96.6345 357.1709 0015989 224.4362 220.2191 16.00930159  9436
    1 43947U 19006A   19083.81116759  .00103815  62739-5  19235-3 0  9995
    2 43947  96.6324 357.2279 0015947 223.4735 218.5865 16.00945210  9446
    1 43947U 19006A   19083.93495759  .00147725  10814-4  27243-3 0  9990
    2 43947  96.6324 357.3507 0015945 222.9977 212.0347 16.00998010  9463
    1 43947U 19006A   19084.04992207  .00154144  11623-4  28342-3 0  9996
    2 43947  96.6354 357.4673 0016254 222.4775 154.7117 16.01036332  9486
    1 43947U 19006A   19084.56994306 -.00389064  56746-4 -71643-3 0  9995
    2 43947  96.6360 357.9786 0015819 222.1151 270.2904 16.00942425  9569
    1 43947U 19006A   19084.68712708  .00107795  66250-5  19877-3 0  9992
    2 43947  96.6366 358.0976 0015957 221.7662 225.5779 16.01024149  9583
    1 43947U 19006A   19084.74556389  .00097765  57838-5  18034-3 0  9991
    2 43947  96.6363 358.1566 0015910 221.6645 202.2537 16.01028963  9598
    1 43947U 19006A   19084.98720666  .00152043  11367-4  27860-3 0  9992
    2 43947  96.6362 358.3958 0015832 220.6056 155.1371 16.01110841  9636
    1 43947U 19006A   19085.19976117  .00022830  21215-5  42802-4 0  9998
    2 43947  96.6349 358.6073 0016094 221.4560 298.5857 16.01085713  9665
    1 43947U 19006A   19085.49952644  .00052661  30351-5  96803-4 0  9996
    2 43947  96.6377 358.9036 0016197 221.8414 224.9544 16.01169686  9718
    1 43947U 19006A   19085.56842683  .00054526  31167-5  10000-3 0  9997
    2 43947  96.6377 358.9720 0016198 221.5803 262.1287 16.01202917  9722
    1 43947U 19006A   19085.80962006  .00050945  29637-5  93510-4 0  9994
    2 43947  96.6379 359.2115 0016283 220.5574 212.4511 16.01198803  9765'''

    mass = 740

    tle_raw = [line.strip() for line in tle_raw.split('\n')]
    if len(tle_raw) % 2 != 0:
        raise Exception('Not even number of lines [not TLE compatible]')
    TLEs = zip(tle_raw[0::2], tle_raw[1::2])

    event_date = np.datetime64('2019-03-27T05:40')
    event_mjd = dpt.npdt2mjd(event_date)

    M_earth = propagator_sgp4.SGP4.GM*1e9/consts.G

    pop = Population(
        name='Simulated Microsat R debris',
        extra_columns = ['A', 'm', 'd', 'C_D', 'C_R'],
        space_object_uses = [True, True, True, True, True],
        propagator = propagator,
        propagator_options = propagator_options,
    )

    pop.allocate(num)


    tle_mjds = np.empty((len(TLEs),), dtype=np.float64)
    for line_id, lines in enumerate(TLEs):
        line1, line2 = lines
        jd0 = tle.tle_jd(line1)
        mjd0 = dpt.jd_to_mjd(jd0)
        tle_mjds[line_id] = mjd0

    best_tle = np.argmin(np.abs(tle_mjds - event_mjd))


    line1, line2 = TLEs[best_tle]
    sat_id = tle.tle_id(line1)
    jd0 = tle.tle_jd(line1)
    mjd0 = dpt.jd_to_mjd(jd0)

    state_TEME = tle.TLE_propagation_TEME(line1, line2, dpt.mjd_to_jd(event_mjd))

    #first has no pert

    states = np.repeat(state_TEME, num, axis=1)

    np.random.seed(seed)

    kep = dpt.cart2kep(state_TEME, m=mass, M_cent=M_earth, radians=False)
    pop.objs[0][1] = kep[0]*1e-3
    pop.objs[0][2] = kep[1]
    pop.objs[0][3] = kep[2]
    pop.objs[0][4] = kep[4]
    pop.objs[0][5] = kep[3]
    pop.objs[0][6] = dpt.true2mean(kep[5], kep[1], radians=False)
        
    pop.objs[0][0] = float(sat_id)
    pop.objs[0][7] = mjd0

    pop.objs[0][8] = np.pi*1.0**2
    pop.objs[0][9] = mass
    pop.objs[0][10] = 2.0
    pop.objs[0][11] = 2.3
    pop.objs[0][12] = 1.0

    ind = 1
    while ind < num:
        C_D = np.random.rand(1)*(C_D_range[1] - C_D_range[0]) + C_D_range[0]
        rho = 11e3
        while rho > 10e3:
            r = np.random.rand(1)*(radii_range[1] - radii_range[0]) + radii_range[0]
            A = np.pi*r**2
            m = np.random.rand(1)*(mass_range[1] - mass_range[0]) + mass_range[0]
            vol = 4.0/3.0*np.pi*r**3
            rho = m/vol

        pert_state = states[:,ind].copy()

        dv = np.random.rand(1)*max_dv
        ddir = _rand_sph(1).T

        pert_state[:3] += ddir[:,0]*dv

        kep = dpt.cart2kep(pert_state, m=mass, M_cent=M_earth, radians=False)

        if kep[0]*(1.0 - kep[1]) > 6353.0e3+100e3:

            pop.objs[ind][1] = kep[0]*1e-3
            pop.objs[ind][2] = kep[1]
            pop.objs[ind][3] = kep[2]
            pop.objs[ind][4] = kep[4]
            pop.objs[ind][5] = kep[3]
            pop.objs[ind][6] = dpt.true2mean(kep[5], kep[1], radians=False)
                    
            pop.objs[ind][0] = float(sat_id) + num
            pop.objs[ind][7] = mjd0

            pop.objs[ind][8] = A
            pop.objs[ind][9] = m
            pop.objs[ind][10] = r*2.0
            pop.objs[ind][11] = C_D
            pop.objs[ind][12] = 1.0

            ind += 1

    if 'out_frame' in pop.propagator_options:
        out_f = pop.propagator_options['out_frame']
    else:
        out_f = 'ITRF'
    pop.propagator_options['out_frame'] = 'TEME'
    pop = propagate_population(pop, mjd)
    pop.propagator_options['out_frame'] = out_f

    return pop


def _get_MicrosatR_state(mjd):
    tle_raw = '''1 43947U 19006A   19085.80962006  .00050945  29637-5  93510-4 0  9994
    2 43947  96.6379 359.2115 0016283 220.5574 212.4511 16.01198803  9765'''

    tle_raw = [line.strip() for line in tle_raw.split('\n')]
    if len(tle_raw) % 2 != 0:
        raise Exception('Not even number of lines [not TLE compatible]')
    line1, line2 = tle_raw[0], tle_raw[1]


    jd0 = tle.tle_jd(line1)
    mjd0 = dpt.jd_to_mjd(jd0)
    sat_id = tle.tle_id(line1)

    state_TEME = tle.TLE_propagation_TEME(line1, line2, dpt.mjd_to_jd(mjd))

    return state_TEME, sat_id

def simulate_Microsat_R_debris_v2(num, max_dv, rho_range, mass_range, seed, propagator, propagator_options, mjd):


    mass = 740

    event_date = np.datetime64('2019-03-27T05:40')
    event_mjd = dpt.npdt2mjd(event_date)

    M_earth = propagator_sgp4.SGP4.GM*1e9/consts.G

    pop = Population(
        name='Simulated Microsat-R debris',
        extra_columns = ['A', 'm', 'd', 'C_D', 'C_R', 'dV', 'dx', 'dy', 'dz'],
        space_object_uses = [True, True, True, True, True, False, False, False, False],
        propagator = propagator,
        propagator_options = propagator_options,
    )

    pop.allocate(num)

    state_TEME, sat_id = _get_MicrosatR_state(event_mjd)

    #first has no pert

    states = np.repeat(state_TEME, num, axis=1)

    np.random.seed(seed)
    popts = copy.deepcopy(propagator_options)
    popts['out_frame'] = propagator_options['in_frame']

    prop = propagator(**popts)

    def mrho2d(m,rho):
        V = m/rho
        return 2.0*(V*3.0/(4.0*np.pi))**(1.0/3.0)

    prop_data = {}

    ind = 0
    trys = 0
    while ind < num:
        trys += 1
        print('try number {}'.format(trys))

        prop_data['C_D'] = 2.3
        prop_data['C_R'] = 1.0
        prop_data['mjd0'] = event_mjd

        rho_v = np.random.rand(1)*(rho_range[1] - rho_range[0]) + rho_range[0]
        m_v = np.random.rand(1)*(mass_range[1] - mass_range[0]) + mass_range[0]
        d_v = mrho2d(m_v, rho_v)

        prop_data['m'] = m_v[0]
        prop_data['d'] = d_v[0]
        prop_data['A'] = np.pi*d_v[0]**2/4.0

        pert_state = state_TEME[:,0].copy()

        dv = np.random.rand(1)*max_dv
        ddir = _rand_sph(1).T

        pert_state[3:] += ddir[:,0]*dv

        prop_data['x'], prop_data['y'], prop_data['z'] = pert_state[:3]
        prop_data['vx'], prop_data['vy'], prop_data['vz'] = pert_state[3:]
        prop_data['t'] = (mjd - event_mjd)*3600.0*24.0

        try:
            prop_pert_state = prop.get_orbit_cart(**prop_data)
        except Exception as e:
            #print(e)
            continue

        print('object added - {} of {}'.format(ind, num))
        kep = dpt.cart2kep(prop_pert_state[:,0], m=mass, M_cent=M_earth, radians=False)

        pop.objs[ind][1] = kep[0]*1e-3
        pop.objs[ind][2] = kep[1]
        pop.objs[ind][3] = kep[2]
        pop.objs[ind][4] = kep[4]
        pop.objs[ind][5] = kep[3]
        pop.objs[ind][6] = dpt.true2mean(kep[5], kep[1], radians=False)
                
        pop.objs[ind][0] = float(sat_id) + num
        pop.objs[ind][7] = mjd

        pop.objs[ind]['A'] = prop_data['A']
        pop.objs[ind]['d'] = prop_data['d']
        pop.objs[ind]['m'] = prop_data['m']
        pop.objs[ind]['C_D'] = prop_data['C_D']
        pop.objs[ind]['C_R'] = prop_data['C_R']

        pop.objs[ind]['dV'] = dv
        pop.objs[ind]['dx'] = ddir[0,0]
        pop.objs[ind]['dy'] = ddir[1,0]
        pop.objs[ind]['dz'] = ddir[2,0]

        ind += 1

    return pop




Microsat_R_debris_raw_tle = '''1 44117U 19006C   19094.64070647  .00455742  26153-4  80334-2 0  9998
2 44117  95.3237   4.1062 0861779 350.0688   8.4347 13.95892635  1153
1 44118U 19006D   19094.46329163  .03050626  31109-2  24862-1 0  9992
2 44118  96.1629   6.2262 0272847 356.3910   3.5361 15.35582568  1229
1 44119U 19006E   19094.46646588  .06439739  12728-5  97304-2 0  9996
2 44119  94.9178   4.6437 0181254  28.4691 332.6337 15.80679644  1248
1 44120U 19006F   19095.00036339  .00527918  52265-4  56871-2 0  9990
2 44120  95.5094   5.4447 0422033 357.9318   2.0147 14.99373378  1301
1 44121U 19006G   19095.01964115  .01508404  50846-3  14651-1 0  9995
2 44121  95.5844   5.6491 0360459 355.4980   4.3038 15.14169030  1318
1 44122U 19006H   19094.75258187  .01086181  27867-3  91501-2 0  9991
2 44122  95.6006   5.5919 0297801 357.2150   2.7421 15.29557898  1264
1 44123U 19006J   19095.06527372  .00842597  15862-3  65277-2 0  9995
2 44123  96.9468   7.8798 0329609 328.4083  29.7731 15.24689115  1321
1 44124U 19006K   19095.01748409  .02829797  25262-2  22519-1 0  9991
2 44124  95.5327   5.6875 0284095 357.7110   2.2814 15.33730995  1318
1 44125U 19006L   19094.82916466  .01599065  62368-3  14033-1 0  9992
2 44125  95.4484   5.4059 0316221 356.8319   3.0912 15.25133571  1273
1 44126U 19006M   19094.58326523  .03116733  28646-2  27747-1 0  9990
2 44126  95.4679   5.2200 0335316 358.4564   3.3570 15.21318838  1238
1 44127U 19006N   19095.03666368  .02187619  17695-2  13453-1 0  9997
2 44127  96.0706   6.8319 0171368 349.2468  10.5144 15.59755513  1133
1 44128U 19006P   19095.07940328  .00382116  42589-4  23241-2 0  9999
2 44128  96.2830   7.3605 0146784 354.4416   5.5208 15.63728289  1155
1 44129U 19006Q   19094.99492147  .01670420  99478-3  93211-2 0  9992
2 44129  96.2485   7.1558 0148156 351.6408   8.2387 15.65407990  1133
1 44130U 19006R   19095.00058036  .00153632  65541-5  14379-2 0  9996
2 44130  96.3346   7.0172 0280946 346.9670  12.4412 15.30818108  1129
1 44131U 19006S   19094.06316062  .00600230  47213-4  10182-1 0  9991
2 44131  95.4882   4.0634 0798088 353.5981   5.5555 14.10377672   757
1 44132U 19006T   19094.73963079  .12979788  19429-5  11454-1 0  9994
2 44132  96.4009   7.4170 0027412 339.9728  20.0480 16.12094571   954
1 44133U 19006U   19095.04008018  .00142556  57788-5  12170-2 0  9992
2 44133  96.4939   7.2182 0324841 329.5554  28.7166 15.23879318   964
1 44134U 19006V   19094.98522109  .00061942  16890-5  96352-3 0  9990
2 44134  96.0918   5.7448 0663376 349.5401   9.2536 14.41154034   907
1 44135U 19006W   19094.87046930  .02680902  37410-2  12244-1 0  9991
2 44135  96.2107   7.1091 0112268 355.6726   4.3559 15.75469894   954
1 44136U 19006X   19095.04109403  .00809162  14876-3  71821-2 0  9995
2 44136  94.5750   4.3171 0296364 354.3855   5.4088 15.28815365   963
1 44137U 19006Y   19094.80053007  .03433409  10222-1  12967-1 0  9991
2 44137  96.1784   7.0131 0082221 347.2816  12.6380 15.84654719   958
1 44138U 19006Z   19095.04584499  .01687024  20239-5  42295-3 0  9997
2 44138  96.6639   8.4300 0125246 256.7733 101.9558 16.11801396  1016
1 44139U 19006AA  19095.01018649  .01863761  14334-2  76060-2 0  9990
2 44139  95.4641   6.0314 0132951 316.5326  42.5520 15.73941483   983
1 44140U 19006AB  19094.99401962  .00937025  14528-3  10253-1 0  9993
2 44140  95.6393   5.3278 0564445   4.3929 356.1942 14.68800488  1288
1 44141U 19006AC  19095.05118892  .00929759  13751-3  12486-1 0  9991
2 44141  96.1001   5.9232 0590442 356.3473   3.3508 14.59478967  1289
1 44142U 19006AD  19095.08823010  .00450387  29805-4  67642-2 0  9999
2 44142  96.5890   6.4539 0677192 345.9062  12.4022 14.38737046  1276
1 44143U 19006AE  19095.05197432  .09474171  20938-5  60107-2 0  9993
2 44143  96.7877   8.4977 0027864 281.4570  78.3589 16.16667874  1307
1 44144U 19006AF  19095.07438984  .00377444  46376-4  18035-2 0  9998
2 44144  95.6350   6.4585 0094274 347.7575  12.1403 15.76217231   645
1 44145U 19006AG  19095.04888338  .01607578  46803-3  17450-1 0  9996
2 44145  95.6463   5.3980 0532778   2.6283 357.7536 14.75786071   597
1 44146U 19006AH  19095.02837073  .00089273  19163-5  14515-3 0  9992
2 44146  96.5911   8.1980 0103814 264.0180  94.9258 15.93838830   630
1 44147U 19006AJ  19095.06147560  .00157958  41062-5  23269-2 0  9996
2 44147  95.7673   5.1413 0774807   1.8709 358.5090 14.17974235   565
1 44148U 19006AK  19095.05278179  .00260163  23632-4  10121-2 0  9995
2 44148  96.6499   8.1178 0099791 308.5716  50.6643 15.79515068   625
1 44149U 19006AL  19095.04985371  .00138584  63554-5  90991-3 0  9997
2 44149  96.1023   6.9822 0182961   3.3819 356.8627 15.56156057   618
1 44150U 19006AM  19094.24965542  .03380564  82179-2  13716-1 0  9997
2 44150  96.1314   6.4062 0109734 347.4104  13.2096 15.79057773   490
1 44151U 19006AN  19095.07776720  .01139469  39308-3  70293-2 0  9993
2 44151  96.1917   7.1013 0175670 351.4915   8.3374 15.58677746   614
1 44152U 19006AP  19095.00425471  .00274120  17165-4  25192-2 0  9990
2 44152  95.0640   5.0844 0281536 351.5372   8.1175 15.31038960   596
1 44153U 19006AQ  19095.10037387  .00335950  24673-4  29766-2 0  9999
2 44153  95.0683   5.1111 0309678 359.8179   0.2901 15.26184379   608
1 44154U 19006AR  19095.03166584  .00440937  19896-4  10013-1 0  9991
2 44154  96.7141   5.2587 1287411 353.6687   4.9420 12.99476955   508
1 44155U 19006AS  19095.04684293  .00311061  27074-4  21063-2 0  9995
2 44155  96.0164   6.8512 0167909 345.7280  13.9264 15.58091161   608
1 44156U 19006AT  19094.69707226  .05553740  38015-1  26327-1 0  9996
2 44156  96.2572   7.0051 0089971 352.8442   7.1537 15.81344557   554
1 44157U 19006AU  19094.99880469  .00367995  17112-4  47305-2 0  9999
2 44157  95.4389   4.4696 0889261   7.5858 353.7787 13.95328540   535
1 44158U 19006AV  19095.06474477  .00180650  98283-5  13018-2 0  9991
2 44158  96.3276   7.3180 0184890 339.8510  19.5511 15.53942281   603
1 44159U 19006AW  19094.21895253  .16047287  19078-5  10135-1 0  9993
2 44159  96.3834   6.9600 0020619 341.3096  18.7434 16.17187137   484
1 44160U 19006AX  19095.08101037  .00234973  69896-5  43180-2 0  9994
2 44160  95.3215   4.1745 0976328 347.1157  10.6586 13.70329689   538
1 44161U 19006AY  19095.04864178  .00114018  53545-5  36847-3 0  9997
2 44161  96.2750   7.3879 0190135 296.2142  61.9683 15.67668399   605
1 44162U 19006AZ  19095.07873983  .00836557  10839-3  11532-1 0  9993
2 44162  96.0404   5.8159 0611774 356.4193   3.2724 14.54410367   561
1 44163U 19006BA  19095.07356566  .01011177  21458-3  99281-2 0  9998
2 44163  95.5073   5.5907 0365152 357.0211   2.8845 15.12958148   434
1 44164U 19006BB  19095.09553700  .00177287  73485-5  19099-2 0  9992
2 44164  96.6780   7.3048 0403399 338.2118  20.2360 15.03284262   436
1 44165U 19006BC  19095.07242078  .00225857  14659-4  14543-2 0  9997
2 44165  96.7326   7.9452 0202955 324.2145  34.5647 15.52942310   445
1 44166U 19006BD  19095.04140141  .00362798  15115-4  72119-2 0  9990
2 44166  95.7142   4.5173 1043123 357.1107   2.4300 13.54482014   381
1 44167U 19006BE  19095.05516230  .01576780  86773-3  88344-2 0  9991
2 44167  95.7015   6.3580 0149222 336.5257  22.9240 15.65131666   441
1 44168U 19006BF  19095.04961209  .00527696  86952-4  28336-2 0  9997
2 44168  96.3473   7.5037 0118347 350.9074   9.0046 15.70583958   449
1 44169U 19006BG  19095.04600491  .00244967  21667-4  83571-3 0  9993
2 44169  97.2645   9.1181 0110262 298.8316  60.1918 15.80462905   294
1 44170U 19006BH  19094.29776842  .01477004  51465-3  13916-1 0  9998
2 44170  94.7415   4.0349 0321201 354.4045   5.3637 15.22743226   167
1 44171U 19006BJ  19095.07665140  .00743869  12904-3  63401-2 0  9998
2 44171  95.6686   6.0475 0278790 356.6778   3.2606 15.33067039   280
1 44172U 19006BK  19093.24832937  .06491989  10561-5  62839-2 0  9998
2 44172  94.5693   3.5604 0110923  40.5090 320.5757 15.99896821    05
1 44173U 19006BL  19093.25107377  .00092895  25285-5  10000-2 0  9992
2 44173  95.4100   4.1032 0410196   3.5176 356.9078 15.01902668    09'''

def Microsat_R_debris_TLE(mjd=None):
    tle_raw = [line.strip() for line in Microsat_R_debris_raw_tle.split('\n')]
    if len(tle_raw) % 2 != 0:
        raise Exception('Not even number of lines [not TLE compatible]')

    TLEs = zip(tle_raw[0::2], tle_raw[1::2])
    pop = tle_snapshot(TLEs, sgp4_propagation=True)

    if mjd is None:
        return pop

    pop.propagator_options['out_frame'] = 'TEME'
    pop = propagate_population(pop, mjd)
    pop.propagator_options['out_frame'] = 'ITRF'
    pop.propagator = propagator_sgp4.PropagatorSGP4

    rem_cols = ['A', 'm', 'd', 'C_D', 'C_R'] + ['line1', 'line2']
    for col in rem_cols:
        pop.space_object_uses[pop.header.index(col)] = False

    pop.add_column(
        'B',
        space_object_uses=True,
    )

    pop['d'] = 0.1

    for ind in range(len(pop)):

        bstar = tle.tle_bstar(pop[ind]['line1'])
        bstar = bstar/(propagator_sgp4.SGP4.R_EARTH*1000.0)
        B = bstar/(0.5*propagator_sgp4.SGP4.RHO0)

        pop[ind]['B'] = B

    return pop

def Microsat_R_debris(mjd, num, radii_range, mass_range, propagator, propagator_options):

    tle_raw = Microsat_R_debris_raw_tle

    tle_raw = [line.strip() for line in tle_raw.split('\n')]
    if len(tle_raw) % 2 != 0:
        raise Exception('Not even number of lines [not TLE compatible]')
    TLEs = zip(tle_raw[0::2], tle_raw[1::2])

    M_earth = propagator_sgp4.SGP4.GM*1e9/consts.G

    pop = Population(
        name='Microsat R debris',
        extra_columns = ['A', 'm', 'd', 'C_D', 'C_R'],
        space_object_uses = [True, True, True, True, True],
        propagator = propagator,
        propagator_options = propagator_options,
    )

    pop.allocate(len(TLEs)*num)

    delete_inds = []
    cnt = 0
    for ind in range(num):
        for line_id, lines in enumerate(TLEs):
            line1, line2 = lines

            sat_id = tle.tle_id(line1)
            jd0 = tle.tle_jd(line1)
            mjd0 = dpt.jd_to_mjd(jd0)

            state_TEME = tle.TLE_propagation_TEME(line1, line2, dpt.mjd_to_jd(mjd))
            kep = dpt.cart2kep(state_TEME, m=0.0, M_cent=M_earth, radians=False)

            if np.any(np.isnan(kep)):
                delete_inds.append(cnt)
                continue

            pop.objs[cnt][1] = kep[0]*1e-3
            pop.objs[cnt][2] = kep[1]
            pop.objs[cnt][3] = kep[2]
            pop.objs[cnt][4] = kep[4]
            pop.objs[cnt][5] = kep[3]
            pop.objs[cnt][6] = dpt.true2mean(kep[5], kep[1], radians=False)
            
            pop.objs[cnt][0] = float(sat_id)
            pop.objs[cnt][7] = mjd

            rho = 1.1e4
            while rho > 1e4 or rho < 1e2:
                r = np.random.rand(1)*(radii_range[1] - radii_range[0]) + radii_range[0]
                A = np.pi*r**2
                m = np.random.rand(1)*(mass_range[1] - mass_range[0]) + mass_range[0]
                vol = 4.0/3.0*np.pi*r**3
                rho = m/vol

            pop.objs[cnt][8] = A
            pop.objs[cnt][9] = m
            pop.objs[cnt][10] = r*2.0
            pop.objs[cnt][11] = 2.3
            pop.objs[cnt][12] = 1.0

            cnt += 1
        
    pop.delete(delete_inds)

    return pop