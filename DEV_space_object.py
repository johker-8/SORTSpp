import space_object as so
import numpy as n
import plothelp
import matplotlib.pyplot as plt

t=n.linspace(0,3*3600,num=1000)

# test propagation
o=so.space_object(a=10000,e=0.1,i=69,raan=0,aop=0,mu0=0,C_D=2.3,A=1.0,m=1.0,oid=1)

print(o)

ecefs=o.get_state(t)

#o2=so.space_object(x=ecefs[0,0], y=ecefs[1,0], z=ecefs[2,0], vx = ecefs[3,0], vy = ecefs[4,0], vz = ecefs[5,0],C_D=2.3,A=1.0,m=1.0,oid=2)
o2=so.space_object(x=o.x, y=o.y, z=o.z, vx = o.vx, vy = o.vy, vz = o.vz,C_D=2.3,A=1.0,m=1.0,oid=2)

print(o2)


ecefs2=o2.get_state(t)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(15, 5)
plothelp.draw_earth_grid(ax)

ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.5,color="black")
ax.plot(ecefs2[0,:],ecefs2[1,:],ecefs2[2,:],".",alpha=0.5,color="blue")
plt.title("Orbital propagation test")
plt.show()


print('Testing update function:')
print(o)
print('\n UPDATING SPACE OBJECT \n')

fig = plt.figure(figsize=(15,12))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(15, 5)
plothelp.draw_earth_grid(ax)

ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.2,color="black")
d_km = 100.0
o.update(x=o.x + d_km)
ecefs=o.get_state(t)
ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.2,color="black")

o.update(x=o.x - 2*d_km)
ecefs=o.get_state(t)
ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.2,color="black")
o.update(x=o.x + d_km)

o.update(y=o.y + d_km)
ecefs=o.get_state(t)
ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.2,color="black")

o.update(y=o.y - 2*d_km)
ecefs=o.get_state(t)
ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.2,color="black")
o.update(y=o.y + d_km)

o.update(z=o.z + d_km)
ecefs=o.get_state(t)
ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.2,color="black")

o.update(z=o.z - 2*d_km)
ecefs=o.get_state(t)
ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.2,color="black")
o.update(z=o.z + d_km)

plt.title("Orbital propagation test PERTURB")
plt.show()




print(o)
