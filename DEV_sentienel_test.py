import numpy as np
from propagator_orekit import PropagatorOrekit
import unittest
import numpy.testing as nt
import dpt_tools as dpt

'''
p = PropagatorOrekit(in_frame='ITRF', out_frame='ITRF')

# Statevector from Sentinel-1 precise orbit (in ECEF frame)
sv = np.array([('2015-04-30T05:45:44.000000000',
    [2721793.785377, 1103261.736653, 6427506.515945],
    [ 6996.001258,  -171.659563, -2926.43233 ])],
      dtype=[('utc', '<M8[ns]'), ('pos', '<f8', (3,)), ('vel', '<f8', (3,))])
x,  y,  z  = [float(i) for i in sv[0]['pos']]
vx, vy, vz = [float(i) for i in sv[0]['vel']]
mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

state = [2721793.785377, 1103261.736653, 6427506.515945] + [ 6996.001258,  -171.659563, -2926.43233 ]
state = np.array(state)

state_kep = dpt.cart2kep(state, m=2300.0, M_cent=p.M_earth, radians=False)

t = [0]
pv = p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0,
                      m=2300., C_R=1., C_D=2.3, A=4*2.3)


pv2 = p.get_orbit(
    t, 
    state_kep[0], state_kep[1], state_kep[2], 
    state_kep[4], state_kep[3], dpt.true2mean(state_kep[5],state_kep[1],radians=False), 
    mjd0,
    m=2300., C_R=1., C_D=2.3, A=4*2.3,
)

print(state - pv2[:,0])
print(state - pv[:,0])
print(pv[:,0] - pv2[:,0])
'''


class TestSentinel(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.p = PropagatorOrekit(in_frame='ITRF', out_frame='ITRF', frame_tidal_effects=True)
        super(TestSentinel, self).__init__(*args, **kwargs)


    def test_tg_cart0(self):
        '''
        See if cartesian orbit interface recovers starting state
        '''
        

        # Statevector from Sentinel-1 precise orbit (in ECEF frame)
        sv = np.array([('2015-04-30T05:45:44.000000000',
            [2721793.785377, 1103261.736653, 6427506.515945],
            [ 6996.001258,  -171.659563, -2926.43233 ])],
              dtype=[('utc', '<M8[ns]'), ('pos', '<f8', (3,)), ('vel', '<f8', (3,))])
        x,  y,  z  = [float(i) for i in sv[0]['pos']]
        vx, vy, vz = [float(i) for i in sv[0]['vel']]
        mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

        t = [0]
        pv = self.p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0,
                              m=2300., C_R=1., C_D=2.3, A=4*2.3)

        print('pos error:')
        dp = sv['pos'] - pv[:3].T
        print('{:.5e}, {:.5e}, {:.5e}'.format(*dp[0].tolist()))

        print('vel error:')
        dv = sv['vel'] - pv[3:].T
        print('{:.5e}, {:.5e}, {:.5e}'.format(*dv[0].tolist()))

        nt.assert_array_almost_equal(sv['pos'] / pv[:3].T, np.ones((1,3)), decimal=7)
        nt.assert_array_almost_equal(sv['vel'] / pv[3:].T, np.ones((1,3)), decimal=7)


    def test_tg_cart6(self):
        '''
        See if cartesian orbit propagation interface matches actual orbit
        '''

        # Statevector from Sentinel-1 precise orbit (in ECEF frame)
        sv = np.array([
            ('2015-04-30T05:45:44.000000000',
                [2721793.785377, 1103261.736653, 6427506.515945],
                [ 6996.001258,  -171.659563, -2926.43233 ]),
            ('2015-04-30T05:45:54.000000000',
                [2791598.832403, 1101432.471307, 6397880.289842],
                [ 6964.872299,  -194.182612, -2998.757484]),
            ('2015-04-30T05:46:04.000000000',
                [2861088.520266, 1099378.309568, 6367532.487662],
                [ 6932.930021,  -216.638226, -3070.746198]),
            ('2015-04-30T05:46:14.000000000',
                [2930254.733863, 1097099.944255, 6336466.514344], 
                [ 6900.178053,  -239.022713, -3142.39037 ]), 
            ('2015-04-30T05:46:24.000000000',
                [2999089.394834, 1094598.105058, 6304685.855646],
                [ 6866.620117,  -261.332391, -3213.681933]),
            ('2015-04-30T05:46:34.000000000', 
                [3067584.462515, 1091873.55841 , 6272194.077798], 
                [ 6832.260032,  -283.563593, -3284.612861])],
            dtype=[('utc', '<M8[ns]'), ('pos', '<f8', (3,)), ('vel', '<f8', (3,))])

        x,  y,  z  = [float(i) for i in sv[0]['pos']]
        vx, vy, vz = [float(i) for i in sv[0]['vel']]
        mjd0 = (sv[0]['utc'] - np.datetime64('1858-11-17')) / np.timedelta64(1, 'D')

        t = 10*np.arange(6)
        pv = self.p.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0,
                              m=2300., C_R=1., C_D=2.3, A=4*2.3)

        nt.assert_array_almost_equal(sv['pos'] / pv[:3].T, np.ones((6,3)), decimal=5)
        nt.assert_array_almost_equal(sv['vel'] / pv[3:].T, np.ones((6,3)), decimal=5)