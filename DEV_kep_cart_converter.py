import numpy as n
import space_object as so
import plothelp
import sys
import matplotlib.pyplot as plt
from scipy.constants import au
import dpt_tools as dpt

o = n.array([25000e3, 0.7, 72.0, 45.0, 45.0, 25.0],dtype=n.float)
o.shape = (6,1)
obj=so.space_object(a=o[0]*1e-3,e=o[1], i=o[2],raan=o[4],aop=o[3],mu0=dpt.true2mean(o[5],o[1],radians=False),C_D=2.3,A=1.0,m=1.0,diam=0.1)

#ecef needs earth rotation as a function of time
from keplerian_sgp4 import gmst
theta = gmst( obj.mjd0 )
eart_rot = dpt.rot_mat_z(-theta)

x = obj.get_state(0.0)

M_e = 5.972e24

x_dpt = dpt.kep2cart(o, m=n.array([1.0]),M_cent=M_e, radians=False)
o_dpt = dpt.cart2kep(x, m=n.array([1.0]),M_cent=M_e, radians=False)

x_dpt[:3] = eart_rot.dot(x_dpt[:3])
x_dpt[3:] = eart_rot.dot(x_dpt[3:])

print('Initial Kepler:',o)
print('Initial Cart as from propagator:',x)
print('Converted Kepler:',o_dpt)
print('Converted Cart:',x_dpt)

#n.testing.assert_almost_equal(x/au,x_dpt/au, decimal=4)
#n.testing.assert_almost_equal(o,o_dpt, decimal=2)

nu = n.mod(n.linspace(0.0,360.0,num=100) + 90.0,360.0)

'''
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
e_test = [0,0.25,0.5,0.7,0.9]
nu_test = n.linspace(0.0,359.0,num=500)
for e in e_test:
    ax.plot(dpt.true2mean(nu_test,e,radians=False),dpt.true2eccentric(nu_test,e,radians=False))
ax.set(xlabel='M',ylabel='E')
plt.show()
'''

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(15, 5)
plothelp.draw_earth_grid(ax)

max_range = o[0]*1.1

print('NU 0: %.2f, E 0: %.2f, M 0: %.2f'%(nu[0],dpt.true2eccentric(nu[0],o[1],radians=False),dpt.true2mean(nu[0],o[1],radians=False))) 

ecefs = n.empty((6,len(nu)))
ecefs2 = n.empty((6,len(nu)))
for i in range(len(nu)):
    o[5] = nu[i]
    obj.mu0 = dpt.true2mean(o[5],o[1],radians=False)
    my_x = obj.get_state(0.0)
    my_x.shape=(6,)
    ecefs[:,i] = my_x
    my_x = dpt.kep2cart(o, m=n.array([1.0]),M_cent=M_e, radians=False)
    my_x.shape=(6,)
    ecefs2[:3,i] = eart_rot.dot(my_x[:3])
    ecefs2[3:,i] = eart_rot.dot(my_x[3:])

ax.plot([0,max_range],[0,0],[0,0],"-k")
ax.plot([0,max_range],[0,0],[0,0],"-k", label='+x')
ax.plot([0,0],[0,max_range],[0,0],"-b", label='+y')
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)
ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],"xk",alpha=0.5,label='Propagated ecefs')
ax.plot(ecefs2[0,:],ecefs2[1,:],ecefs2[2,:],".r",alpha=0.5,label='Converted ecefs')
ax.plot([ecefs[0,0]],[ecefs[1,0]],[ecefs[2,0]],"ob",alpha=1,label='Propagated ecefs START')
ax.plot([ecefs2[0,0]],[ecefs2[1,0]],[ecefs2[2,0]],"oy",alpha=1,label='Converted ecefs START')
ax.plot([ecefs[0,10]],[ecefs[1,10]],[ecefs[2,10]],"oc",alpha=1,label='Propagated ecefs motion')
ax.plot([ecefs2[0,10]],[ecefs2[1,10]],[ecefs2[2,10]],"oc",alpha=1,label='Converted ecefs motion')

ax.legend()
plt.title("Orbital conversion tests")
#plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
ax.plot(n.sqrt(n.sum(ecefs[:3,:]**2,axis=0)),n.sqrt(n.sum(ecefs[3:,:]**2,axis=0)),"xk",alpha=0.5,label='Propagated ecefs')
ax.plot(n.sqrt(n.sum(ecefs2[:3,:]**2,axis=0)),n.sqrt(n.sum(ecefs2[3:,:]**2,axis=0)),".r",alpha=0.5,label='Converted ecefs')
ax.legend()
plt.show()