#!/usr/bin/env python

import numpy as n
import h5py
import matplotlib.pyplot as plt
import population_library as plib
from scipy.stats import norm
import matplotlib.mlab as mlab

m = plib.master_catalog()

plt.loglog(m["d"],m["A"]/m["m"],".",alpha=0.1)
plt.title("Diameter vs. area/mass\nMASTER 2009")
plt.xlabel("Diameter (m)")
plt.ylabel("Area-to-mass ratio (m$^2$/kg)")
plt.tight_layout()
plt.savefig("report_plots/am_area_to_mass_vs_diam.png")
plt.show()

am=n.log10(m["A"]/m["m"])
n,bins,patches = plt.hist(am,bins=n.linspace(-3,2,num=100),normed=1)


# best fit of data
(mu, sigma) = norm.fit(am)

## the histogram of the data
#n, bins, patches = plt.hist(datos, 60, normed=1, facecolor='green', alpha=0.75)

# add a 'best fit' line
y = mlab.normpdf( bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

plt.title("Area-to-mass histogram\nMASTER 2009")

plt.text(0,0.5,"$\mu=%1.2f$, $\sigma=%1.2f$"%(mu,sigma))
plt.xlabel("Area-to-mass ratio (log$_{10}$ m$^2$/kg)")
plt.tight_layout()
plt.savefig("report_plots/am_area_to_mass_hist.png")
plt.show()

plt.semilogy(m["a"],m["A"]/m["m"],".",alpha=0.1)
plt.xlim([6371+100,6371+2000.0])
plt.title("Apogee vs A/m\nMASTER 2009")
plt.xlabel("Apogee (km)")
plt.ylabel("Area-to-mass ratio (m$^2$/kg)")
plt.tight_layout()
plt.savefig("report_plots/am_apogee_vs_diam.png")
plt.show()
plt.close()
plt.clf()


plt.semilogy((1.0-m["e"])*m["a"]-6371,m["A"]/m["m"],".",alpha=0.1)
plt.xlim([100,2000.0])
plt.title("Apogee vs A/m\nMASTER 2009")
plt.xlabel("Minimum altitude (km)")
plt.ylabel("Area-to-mass ratio (m$^2$/kg)")
plt.tight_layout()
plt.savefig("report_plots/am_apogee_vs_diam.png")
plt.show()
plt.close()
plt.clf()


