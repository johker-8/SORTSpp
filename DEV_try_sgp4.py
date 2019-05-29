import dpt_tools as dpt
import propagator_sgp4
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as consts
M_earth = propagator_sgp4.SGP4.GM*1e9/consts.G

prop = propagator_sgp4.PropagatorSGP4()

def get_orbit(o):
    #print('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*o.tolist()))
    ecef = prop.get_orbit(
        t=0.0,
        a=o[0],
        e=o[1],
        inc=o[2],
        raan=o[4],
        aop=o[3],
        mu0=dpt.true2mean(o[5],o[1],radians=False),
        mjd0=dpt.jd_to_mjd(2457126.2729),
        C_D=2.3,
        A=1.0,
        m=1.0,
    )
    return ecef

#dpt.plot_ref_orbit(get_orbit)
dpt.plot_ref_orbit(get_orbit, orb_init = np.array([10000e3, 0, 0, 0, 0], dtype=np.float))
dpt.plot_ref_orbit(get_orbit, orb_init = np.array([10000e3, 0.1, 0, 0, 0], dtype=np.float))
dpt.plot_ref_orbit(get_orbit, orb_init = np.array([10000e3, 0.1, 0.1, 0, 0], dtype=np.float))

def get_orbit_cart(o):
    cart = dpt.kep2cart(o, m=1.0, M_cent=M_earth, radians=False)
    #print('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*cart.tolist()))
    ecef = prop.get_orbit_cart(
        t=0.0,
        x=cart[0],
        y=cart[1],
        z=cart[2],
        vx=cart[3],
        vy=cart[4],
        vz=cart[5],
        mjd0=dpt.jd_to_mjd(2457126.2729),
        C_D=2.3,
        A=1.0,
        m=1.0,
    )
    return ecef


#dpt.plot_ref_orbit(get_orbit_cart)

'''
init_data = {
            'a': 7500e3,
            'e': 0,
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


import plothelp

ecef = prop.get_orbit(
    t=np.linspace(0,3600.0*24,num=10000,dtype=np.float),
    **init_data
)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(15, 5)
plothelp.draw_earth_grid(ax)

ax.plot(ecef[0,:],ecef[1,:],ecef[2,:],"-k",alpha=0.5)

plt.show()

'''