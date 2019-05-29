#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import debris
import radar_library as rl

radars=[]
radars.append(rl.eiscat_uhf())
radars.append(rl.tromso_space_radar())
radars.append(rl.eiscat_svalbard())
radars.append(rl.eiscat_3d_module())
radars.append(rl.eiscat_3d())

range_m=10**n.linspace(5,8,num=100)

for r in radars:
    min_diam=n.zeros(len(range_m))
    gain_tx=r._tx[0].beam.I_0
    gain_rx=r._rx[0].beam.I_0
    rx_noise=r._tx[0].rx_noise
    B=r._tx[0].coh_int_bandwidth
    wavelength=r._tx[0].wavelength
    tx_power=r._tx[0].tx_power
    print(r._tx[0])

    for ri,rng_m in enumerate(range_m):
        min_diam[ri]=debris.target_diameter(gain_tx, gain_rx, wavelength, tx_power, rng_m, rng_m, enr=1.0, bandwidth=B, rx_noise_temp=rx_noise)
    plt.loglog(range_m/1e3,min_diam,label=r.name)
    
plt.xlabel("Range (km)")
plt.ylabel("System noise equivalent diameter (m)")
plt.legend()
plt.tight_layout()
plt.savefig("report_plots/system_noise_equiv_diameter.png")
plt.show()


exit(0)
