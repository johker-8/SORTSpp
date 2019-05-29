import numpy as n
import plothelp
import matplotlib.pyplot as plt
from scipy.constants import au
import dpt_tools as dpt

M_e = 5.972e24

def plot_orb_conv(orb_init, res=100):
    o = n.empty((6,res),dtype=n.float)
    nu = n.linspace(0, 360.0, num=res, dtype=n.float)

    for i in range(res):
        o[:5,i] = orb_init
        o[5,i] = nu[i]

    x_dpt = dpt.kep2cart(o, m=n.array([1.0]), M_cent=M_e, radians=False)

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, 5)
    plothelp.draw_earth_grid(ax)

    max_range = orb_init[0]*1.1

    ax.plot([0,max_range],[0,0],[0,0],"-k")
    ax.plot([0,max_range],[0,0],[0,0],"-k", label='+x')
    ax.plot([0,0],[0,max_range],[0,0],"-b", label='+y')
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.plot(x_dpt[0,:],x_dpt[1,:],x_dpt[2,:],".k",alpha=0.5,label='Converted elements')
    ax.plot([x_dpt[0,0]],[x_dpt[1,0]],[x_dpt[2,0]],"or",alpha=1,label='$\\nu = 0$')
    ax.plot([x_dpt[0,int(res//4)]],[x_dpt[1,int(res//4)]],[x_dpt[2,int(res//4)]],"oy",alpha=1,label='$\\nu = 0.5\pi$')

    ax.legend()
    plt.title("Kep -> cart: a={} km, e={}, inc={} deg, omega={} deg, Omega={} deg ".format(
        orb_init[0]*1e-3,
        orb_init[1],
        orb_init[2],
        orb_init[3],
        orb_init[4],
    ))
    
'''
a = 50000e3
orb_init = n.array([a, 0, 0.0, 0.0, 0.0], dtype=n.float)
plot_orb_conv(orb_init)

orb_init = n.array([a, 0.8, 0.0, 0.0, 0.0], dtype=n.float)
plot_orb_conv(orb_init)

orb_init = n.array([a, 0.8, 45.0, 0.0, 0.0], dtype=n.float)
plot_orb_conv(orb_init)

orb_init = n.array([a, 0.8, 45.0, 90.0, 0.0], dtype=n.float)
plot_orb_conv(orb_init)

orb_init = n.array([a, 0.8, 45.0, 90.0, 90.0], dtype=n.float)
plot_orb_conv(orb_init)

plt.show()
'''

#detect orb convetions SGP4
from propagator_sgp4 import PropagatorSGP4 as sgp_p
from propagator_orekit import PropagatorOrekit as orekit_p

outf = 'TEME'
#outf = 'ITRF'

prop2 = orekit_p(in_frame='TEME', out_frame=outf)
model = 'full_numerical'

#prop2 = orekit_p(in_frame='TEME', out_frame=outf, earth_gravity='Newtonian', radiation_pressure=False, solarsystem_perturbers=[])
#model = 'simple'

prop = sgp_p(out_frame=outf)


mjd0 = dpt.jd_to_mjd(2457126.2729)


o = n.array([7500e3, 0.05, 45, 90, 90, 123], dtype=n.float)
init_data = {
    'a': o[0],
    'e': o[1],
    'inc': o[2],
    'raan': o[4],
    'aop': o[3],
    'mu0': dpt.true2mean(o[5], o[1], radians=False),
    'mjd0': mjd0,
    'C_D': 2.3,
    'C_R': 1,
    'm': 8000,
    'A': 1.0,
}
t = n.linspace(0,5*3600,num=300,dtype=n.float)
state = prop.get_orbit(t, **init_data)
state2 = prop2.get_orbit(t, **init_data)

dstate = state - state2

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(211)
ax.plot(t/3600.0, n.sqrt(n.sum(dstate[:3,:]**2, axis=0))*1e-3)
ax.set_ylabel('position diff [km]')
ax.set_xlabel('time [h]')
ax.set_title('SGP4 vs Orekit difference [out frame = {}, orekit model = {}]'.format(outf, model))

ax = fig.add_subplot(212)
ax.plot(t/3600.0, n.sqrt(n.sum(dstate[3:,:]**2, axis=0)))
ax.set_ylabel('velocity diff [m/s]')
ax.set_xlabel('time [h]')

plt.savefig('/home/danielk/IRF/E3D_PA/orekit_sgp4_comp/state_diff_{}_{}.png'.format(outf, model))

import plothelp

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(15, 5)
plothelp.draw_earth_grid(ax)

ax.plot(state[0,:],state[1,:],state[2,:],"xb",alpha=0.3,label='SGP4')
ax.plot(state2[0,:],state2[1,:],state2[2,:],"or",alpha=0.1,label='Orekit')

plt.savefig('/home/danielk/IRF/E3D_PA/orekit_sgp4_comp/orbit_3d_{}_{}.png'.format(outf, model))
plt.legend()
plt.show()

def get_orbit(o):
    init_data = {
        'a': o[0],
        'e': o[1],
        'inc': o[2],
        'raan': o[4],
        'aop': o[3],
        'mu0': dpt.true2mean(o[5], o[1], radians=False),
        'mjd0': mjd0,
        'C_D': 2.3,
        'C_R': 1,
        'm': 8000,
        'A': 1.0,
    }
    state = prop.get_orbit(n.array([0.0],dtype=n.float), **init_data)
    return state


def get_orbit2(o):
    init_data = {
        'a': o[0],
        'e': o[1],
        'inc': o[2],
        'raan': o[4],
        'aop': o[3],
        'mu0': dpt.true2mean(o[5], o[1], radians=False),
        'mjd0': mjd0,
        'C_D': 2.3,
        'C_R': 1,
        'm': 8000,
        'A': 1.0,
    }
    state = prop2.get_orbit(n.array([0.0],dtype=n.float), **init_data)
    return state

#dpt.plot_orbit_convention(get_orbit)
#dpt.plot_orbit_convention(get_orbit2)


'''
CONCLUSION: SGP4 uses same conventions as in dpt_tools.kep2cart and it works perfectly
Orekit uses the same convention

BUT there is something wrong when the space object is used????
'''