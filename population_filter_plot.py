#!/usr/bin/env python

import radar_library as rl     # radar network definition
import population_library as plib
import matplotlib.pyplot as plt 

e3d = rl.eiscat_3d(beam='interp', stage=1)
e3d_module = rl.eiscat_3d_module(beam = 'gauss')

# space object population
pop_e3d = plib.filtered_master_catalog_factor(
    radar = e3d,   
    treshhold = 0.01,    # min diam
    min_inc = 50,
    detectability_file="master/celn_20090501_00_EISCAT_3D_PropagatorSGP4_10SNRdB.h5",
    prop_time = 48.0,
)
# space object population
pop_e3d_mod = plib.filtered_master_catalog_factor(
    radar = e3d_module,   
    treshhold = 0.01,    # min diam
    min_inc = 50,
    detectability_file="master/celn_20090501_00_EISCAT_3D_module_PropagatorSGP4_10SNRdB.h5",
    prop_time = 48.0,
)

module=False
if module:
    name="E3D Module"
    fname="e3d_mod"
    pop=pop_e3d_mod
    radar=e3d_module
else:
    name="E3D Stage 1"
    fname="e3d"
    pop=pop_e3d
    radar=e3d
    
a=[]
i=[]
e=[]
d=[]

for o in pop.object_generator():
    a.append(o.a)
    d.append(o.d)
    i.append(o.i)
    e.append(o.e)
n_objects=len(a)
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.semilogx(a,i,".")
plt.xlabel("Apogee (km)")
plt.ylabel("Inclination (deg)")
plt.title(name)
plt.subplot(222)
plt.loglog(a,d,".")
plt.xlabel("Apogee (km)")
plt.ylabel("Diameter (m)")
plt.title("n_detectable=%d"%(n_objects))

plt.subplot(223)
plt.loglog(a,e,".")
plt.xlabel("Apogee (km)")
plt.ylabel("Eccentricity")

plt.subplot(224)
plt.loglog(e,d,".")
plt.xlabel("Eccentricity")
plt.ylabel("Diameter (m)")
plt.tight_layout()
plt.savefig("report_plots/pop_filter_%s.png"%(fname))
plt.show()


