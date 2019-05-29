import space_object as so
import population as p
import plothelp
import numpy as n
import matplotlib.pyplot as plt

t=n.linspace(0,5*24*3600,num=10000,dtype=n.float)

# test propagation
#load the input population
m = p.filtered_master_catalog_factor(None,treshhold=1e-2,seed=12345,filter_name='e3d_planar_beam')

o = m.get_object(15866)

print(o)

ecefs=o.get_state(t)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(15, 5)
plothelp.draw_earth_grid(ax)

ax.plot(ecefs[0,:],ecefs[1,:],ecefs[2,:],".",alpha=0.5,color="black")
plt.title("Orbital propagation test ID {}".format(o.oid))
plt.show()
