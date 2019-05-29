'''Make sure that the get_orbit function implemented 
for the different propagators conform to the standards 
set out by our own orbit conversion system.

We here assume that the earth rotation can be modeled by the gmst function

'''


import numpy as n


from keplerian_sgp4 import sgp4_propagator as sgp_p
import dpt_tools as dpt

init_data = {
    'a': 7000,
    'e': 0.0,
    'inc': 90.0,
    'raan': 10,
    'aop': 10,
    'mu0': 40.0,
    'mjd0': 2457126.2729,
    'C_D': 2.3,
    'm': 8000,
    'A': 1.0,
}

self.t0 = n.array([0.0], dtype=n.float)

p = sgp_p()

ecefs = p.get_orbit(self.t, **self.init_data)
o = n.array([7000e3, 0.0, 90.0, 10.0, 10.0, dpt.mean2true(40.0, 0.0, radians=False)])

x = dpt.kep2cart(o, m=init_data['m'], M_cent=5.98e24, radians=False)

print(x)
print(ecefs)