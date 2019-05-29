import numpy as np
import matplotlib.pyplot as plt

import antenna_library as alib
from antenna import plot_gain_heatmap, plot_gains
import antenna

'''
beam = alib.cassegrain_beam(az0=0.0, el0=90.0, I_0=10**4.5, f=930e6, a0=40, a1=20)

from scipy import interpolate
import matplotlib.pyplot as plt

res = 1000
min_el = 0.0

kx=np.linspace(
    -np.cos(min_el*np.pi/180.0),
    np.cos(min_el*np.pi/180.0),
    num=res,
)
ky=np.linspace(
    -np.cos(min_el*np.pi/180.0),
    np.cos(min_el*np.pi/180.0),
    num=res,
)

S=np.zeros((res,res))
Xmat=np.zeros((res,res))
Ymat=np.zeros((res,res))
for i,x in enumerate(kx):
    for j,y in enumerate(ky):
        z2 = x**2 + y**2
        if z2 < np.cos(min_el*np.pi/180.0)**2:
            k=np.array([x, y, np.sqrt(1.0 - z2)])
            S[i,j]=beam.gain(k)
        else:
            S[i,j] = 0;
        Xmat[i,j]=x
        Ymat[i,j]=y


f = interpolate.interp2d(kx, ky, S, kind='linear')
plt.contourf(Xmat, Ymat, np.log10(S)*10.0)
plt.show()

min_el = 80
res = 500
xnew = np.linspace(
    -np.cos(min_el*np.pi/180.0),
    np.cos(min_el*np.pi/180.0),
    num=res,
)
ynew = np.linspace(
    -np.cos(min_el*np.pi/180.0),
    np.cos(min_el*np.pi/180.0),
    num=res,
)

xx2, yy2 = np.meshgrid(xnew, ynew)
znew = xx2.copy()
for i in range(len(xnew)):
    for j in range(len(ynew)):
        znew[i,j] = f(xx2[i,j], yy2[i,j])

plt.contourf(xx2, yy2, np.log10(znew)*10.0)
plt.show()
exit()
'''
'''
beam_full = alib.e3d_array_beam(az0=0, el0=90.0, I_0=10**4.5)
beam_interp = alib.e3d_array_beam_interp(az0=0, el0=90.0, I_0=10**4.5, res=1000)

res = 100
plot_gain_heatmap(beam_full, res=res, min_el = 75.)
plot_gain_heatmap(beam_interp, res=res, min_el = 75.)
plt.show()

exit()

beam_full = alib.e3d_array_beam_stage1(az0=0, el0=90.0, I_0=10**4.2, opt='dense')
beam_interp = alib.e3d_array_beam_stage1_dense_interp(az0=0, el0=90.0, I_0=10**4.2, res=1000)

res = 100
plot_gain_heatmap(beam_full, res=res, min_el = 75.)
plot_gain_heatmap(beam_interp, res=res, min_el = 75.)
plt.show()

exit()
'''
'''
beam_interp = alib.e3d_array_beam_stage1_dense_interp(az0=0.0, el0=89.0, I_0=10**4.2, res=1000)
beam_interp2 = alib.e3d_array_beam_interp(az0=0, el0=90.0, I_0=10**4.5, res=1000)
print(beam_interp.gain(np.array([0,0,1])))
beam_interp.point_k0(np.array([0,0.5,1]))
print(beam_interp.gain(np.array([0,0.5,1])))

print(beam_interp2.gain(np.array([0,0,1])))
beam_interp2.point_k0(np.array([0,0.5,1]))
print(beam_interp2.gain(np.array([0,0.5,1])))

exit()
'''

'''
beam1 = alib.e3d_array_beam_interp(az0=45, el0=30.0, I_0=10**4.5)
beam2 = alib.e3d_array_beam(az0=45, el0=30.0, I_0=10**4.5)

fig = plt.figure(figsize=(15,7))

ax = fig.add_subplot(121)
plot_gain_heatmap(
    beam1,
    res=100,
    min_el = 70.,
    ax = ax,
    title='Interpolated',
    title_size = 20,
)

ax = fig.add_subplot(122)
plot_gain_heatmap(
    beam2,
    res=100,
    min_el = 70.,
    ax = ax,
    title='Array summation',
    title_size = 20,
)

plt.show()
exit()
'''

'''
beam = alib.e3d_array_beam_interp(az0=0, el0=90.0, I_0=10**4.5)

fig = plt.figure(figsize=(15,7))

az0 = [0, 0, 0, 45, 45, 45]
el0 = [90, 60, 30, 90, 60, 30]

for ind in range(6):
    ax = fig.add_subplot(231 + ind)
    beam.point(az0[ind], el0[ind])
    _, ax = plot_gain_heatmap(
        beam,
        res=100,
        min_el = 70.,
        ax = ax,
        title='Pointing - Az: {} deg, El: {} deg'.format(az0[ind], el0[ind]),
    )

plt.show()
exit()
'''
'''
plot_gain_heatmap(alib.vhf_beam(el0 = 60,f = 1.2e9), res=500, min_el = 75.)
plt.show()
exit()
'''

beam = alib.tsr_fence_beam(f = 1.2e9)


plot_gain_heatmap(beam, res=601, min_el = 0)

plt.show()
exit()


beam = alib.tsr_beam(el0 = 60,f = 1.2e9)

beam.point(az0 = 0.0, el0 = 90.0)
plot_gains([beam], min_el = 88., res=5000)

beam.point(az0 = 90.0, el0 = 90.0)
plot_gains([beam], min_el = 88., res=5000)

for el in [30., 90.]:
    beam.point(az0 = 0.0, el0 = el)
    plot_gain_heatmap(beam, res=301, min_el = 85.0)

plt.show()
exit()



beams = []
els = []
el0 = 90.0
while el0 > 32.0:
    beam = alib.e3d_array_beam_stage1_dense_interp(az0=0, el0=el0, I_0=10**4.2)
    beam.beam_name = 'E3D scan @ elevation: {:.2f}'.format(el0)
    beams.append(beam)
    G = lambda elv: np.array([beam.gain(np.array([0.0, np.cos(np.radians(el)), np.sin(np.radians(el))])) for el in elv])
    angs = np.linspace(30, el0, num=10000, dtype=np.float64)
    g_list = 10.0*np.log10(G(angs))
    ind_g_max = np.argmax(g_list)
    g_max = g_list[ind_g_max]
    ind = np.argmin( np.abs(g_list - (g_max-3.0)) )
    
    els.append(el0)

    width = (angs[ind_g_max] - angs[ind])*2.0
    print('BEAM WIDTH:', width)
    print(g_max, width, angs[ind_g_max], angs[ind], g_list[ind_g_max], g_list[ind])
    el0 -= width

print(els)
print('num: ', len(els))


plot_gains(beams, min_el = 0.)
plt.show()
exit()


for el in [60., 90.]:
    beam.point(az0 = 0.0, el0 = el)
    plot_gain_heatmap(beam, res=201, min_el = 30.)

for el in [60., 90.]:
    beam.point(az0 = 90.0, el0 = el)
    plot_gain_heatmap(beam, res=301, min_el = 85.)



plt.show()
exit()



beams = [
    alib.e3d_array_beam_interp(az0=0, el0=el, I_0=10**4.5)
    for el in np.linspace(90.0, 30.0, num=4)
]

antenna.plot_gains(beams,res=1000,min_el = 0.0)

plt.show()
exit()

beams = [
    alib.planar_beam(az0=0., el0=90., I_0=4.5**10, f=366e6, a0=40., az1=0., el1=90.),
    alib.cassegrain_beam(az0=0., el0=90., I_0=4.5**10, f=940e6, a0=40., a1=20.),
    alib.uhf_beam(az0=0., el0=90., I_0=4.5**10, f=266e6),
    alib.e3d_array_beam_stage1(az0=0, el0=90.0, I_0=10**4.2),
    alib.e3d_array_beam_stage1(az0=0, el0=90.0, I_0=10**4.2, opt='sparse'),
    alib.e3d_array_beam(az0=0, el0=90.0, I_0=10**4.5),
]

#test for peak gain on axis is max
#test for lambda 3db loss

for beam in beams:
    plot_gain_heatmap(beam, res=100, min_el = 75.)

plt.show()