#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import debris
import e3d

range_m=n.linspace(300e3,45000e3,num=1000)

min_diam_e3d=n.zeros(len(range_m))
txgain_e3d=e3d.e3d["tx"][0]["gain"]
txpwr_e3d=e3d.e3d["tx"][0]["txp"]
rxgain_e3d=e3d.e3d["rx"][0]["gain"]
rx_noise_e3d=e3d.e3d["rx"][0]["rx_noise"]
fmf_bandwidth_e3d=e3d.e3d["rx"][0]["bandwidth"]
wavelength_e3d=e3d.e3d["wavelength"]

min_diam_u=n.zeros(len(range_m))
txgain_u=e3d.uhf["tx"][0]["gain"]
txpwr_u=e3d.uhf["tx"][0]["txp"]
rxgain_u=e3d.uhf["rx"][0]["gain"]
rx_noise_u=e3d.uhf["rx"][0]["rx_noise"]
fmf_bandwidth_u=e3d.uhf["rx"][0]["bandwidth"]
wavelength_u=e3d.uhf["wavelength"]

min_diam_e=n.zeros(len(range_m))
min_diam_ges=n.zeros(len(range_m))
txgain_e=e3d.esr["tx"][0]["gain"]
txpwr_e=e3d.esr["tx"][0]["txp"]
rxgain_e=e3d.esr["rx"][0]["gain"]
rx_noise_e=e3d.esr["rx"][0]["rx_noise"]
fmf_bandwidth_e=e3d.esr["rx"][0]["bandwidth"]
wavelength_e=e3d.esr["wavelength"]

for ri,r in enumerate(range_m):
    min_diam_ges[ri]=debris.target_diameter(10**3.1, 10**3.1, 0.20, 300e3, r, r, enr=1.0, bandwidth=50.0, rx_noise_temp=100.0)
    
    min_diam_e3d[ri]=debris.target_diameter(txgain_e3d, rxgain_e3d, wavelength_e3d, txpwr_e3d, r, r, enr=1.0, bandwidth=50, rx_noise_temp=rx_noise_e3d)
    min_diam_u[ri]=debris.target_diameter(txgain_u, rxgain_u, wavelength_u, txpwr_u, r, r, enr=1.0, bandwidth=50, rx_noise_temp=rx_noise_u)
    min_diam_e[ri]=debris.target_diameter(10**6.4, 10**6.4, 0.7, 1e6, r, r, enr=1.0, bandwidth=2272.0/10.0, rx_noise_temp=100.0)        
plt.loglog(range_m/1e3,min_diam_e3d,label="E3D")
plt.loglog(range_m/1e3,min_diam_ges,label="TRA")
plt.loglog(range_m/1e3,min_diam_u,label="UHF")
plt.loglog(range_m/1e3,min_diam_e,label="Arecibo 430 MHz")
plt.xlabel("Range (km)")
plt.ylabel("Noise equivalent diameter (m)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("noise_equiv_diam.png")
plt.show()
