#!/usr/bin/env python

import numpy as n
import h5py
import matplotlib.pyplot as plt
import population_library as plib


"""
  Plot hard target filter statistics
"""
hard_target_lim=40.0
h=h5py.File("master/EISCAT_3D_filter.h5","r")

mo = plib.master_catalog()
m = plib.master_catalog()
m.filter('i', lambda x: x >= 50.0)

det=n.copy(h["detectable"].value)
snr=n.copy(h["peak_snr"].value)

h.close()




def plot_detectable_fraction():
    count=0.0
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.loglog(mo.objs["a"],mo.objs["e"],".",label="All")
    plt.loglog(m.objs["a"][det],m.objs["e"][det],".",label="Detectable")
    plt.legend()
    plt.xlabel("Apogee (km)")
    plt.ylabel("Eccentricity")
    
    plt.subplot(222)
    plt.semilogx(mo.objs["a"],mo.objs["i"],".")
    plt.semilogx(m.objs["a"][det],m.objs["i"][det],".")
    plt.xlabel("Apogee (km)")
    plt.ylabel("Inclination (deg)")
    
    plt.subplot(223)
    plt.loglog(mo.objs["a"],mo.objs["d"],".")
    plt.loglog(m.objs["a"][det],m.objs["d"][det],".")
    plt.xlabel("Apogee (km)")
    plt.ylabel("Diameter (m)")
    
    plt.subplot(224)
    plt.loglog(mo.objs["e"],mo.objs["d"],".")
    plt.loglog(m.objs["e"][det],m.objs["d"][det],".")
    plt.xlabel("Eccentricity")
    plt.ylabel("Diameter (m)")
    plt.tight_layout()
    plt.savefig("report_plots/detectable_scatter.png")
    plt.close()
    plt.clf()
    

    for oi,o in enumerate(m.objs):
        if det[oi]:
            count+=m.objs["Factor"][oi]
        print(count)

        

def plot_hard_target_filter():
#    det=n.copy(h["detectable"].value)
 #   snr=n.copy(h["peak_snr"].value)
    print(m["Factor"][det])
    print(snr.shape)
    print(snr[det][:,0,0])

    filtered=n.where(snr[:,0,0] >= 1e4)[0]
    not_filtered=n.where(snr[:,0,0] <= 1e4)[0]

    plt.hist(10.0*n.log10(snr[filtered,0,0]),weights=m["Factor"][filtered],bins=n.linspace(0,100,num=100),label="Hard target filtered")
    plt.hist(10.0*n.log10(snr[not_filtered,0,0]),weights=m["Factor"][not_filtered],bins=n.linspace(0,100,num=100),label="Not filtered")


    n_total=n.sum(m["Factor"][det])
    n_filtered=n.sum(m["Factor"][filtered])
    plt.title("EISCAT 3D (full system)\ndetectable objects=%d, filtered objects=%d"%(n_total, n_filtered))
    plt.xlabel("Signal-to-noise ratio (dB)")
    plt.ylabel("Count")    
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_plots/hard_target_filter_hist.png")
    plt.close()
    plt.clf()
    

plot_hard_target_filter()
plot_detectable_fraction()
#mo=m.objs
#md=m.objs[det]


#mo=n.array(mo)
#md=n.array(md)
#print(mo)


#print(mo.shape)
#print(md.shape)

