#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import h5py
import population as p


mc= p.master_catalog()
radar_config="eiscat_3d_module"
master_file="master/celn_20090501_00.sim"

h=h5py.File("master/%s_filter.h5"%(radar_config),"r")
master_raw = n.genfromtxt(master_file)
print(h.keys())
detectable=n.copy(h["detectable"].value)
peak_snr=n.copy(h["peak_snr"].value)
h.close()

n_master=master_raw.shape[0]

n_dets=0

snrs=[]
diams=[]
incs=[]
apogs=[]

for i in range(n_master):
    o = mc.get_object(i)
    factor=mc._factor[i]
    diam=o.diam
    
    if detectable[i] > 0.0:

        apogs.append(o.a)
        diams.append(diam)
        snrs.append(n.max(peak_snr[i,:]))
        
        print("%d diam %1.2g factor %1.2f"%(i,o.diam,factor))
        print(detectable[i])
        n_dets+=factor*detectable[i]
        print(n_dets)

plt.hist(10.0*n.log10(snrs),bins=50)
plt.show()
plt.hist(n.log10(apogs),bins=n.log10(n.linspace(300,3000,num=50)+6371))
plt.show()
plt.hist(n.log10(diams),bins=50)
plt.show()
