#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import glob
import h5py


def read_master(fname="master/pop_20180201_00_1cm.pop"):
    a=n.genfromtxt(fname,skip_header=4)
    idx=n.argsort(a[:,3])
    idx=idx[::-1]
    m={"id":a[idx,0],"factor":a[idx,1],"mass":a[idx,2],"diam":a[idx,3],"m/A":a[idx,4],"a":a[idx,5],"e":a[idx,6],"i":a[idx,7],"raan":a[idx,8],"aop":a[idx,9],"m":a[idx,10]}
    return(m)

m=read_master()
mk={}
for i in range(len(m["id"])):
    mk[m["id"][i]]=i

dname="proof_uhf_norad"

fl=glob.glob("%s/det*.h5"%(dname))
fl.sort()

oids=[]
ranges=[]
range_rates=[]
snrs=[]
angles=[]
diams=[]
times=[]
for f in fl:
    h=h5py.File(f,"r")
    print(h.keys())
    if h["range"].value/1e3/2 < 3000.0:
        ranges.append(h["range"].value)
        oids.append(h["oid"].value)
        diams.append(m["diam"][mk[h["oid"].value]])
        range_rates.append(h["range_rate"].value)
        snrs.append(h["snr"].value)
        angles.append(h["on_axis_angle"].value)        
        times.append(h["time"].value)    
    h.close()

print( (n.max(times)-n.min(times))/3600.0)

plt.hist(n.log10(diams),bins=10)
plt.xlabel("Object diameter ($\log_{10}$ meters)")
plt.ylabel("Count")
plt.title("Catalog correlated diameters (Tromso)")
plt.show()
ranges=n.array(ranges)
times=n.array(times)
range_rates=n.array(range_rates)
snrs=n.array(snrs)
print("plotting")
angles=n.array(angles)
oids=n.array(oids)

ho=h5py.File("%s/all_%s.h5"%(dname,dname),"w")
ho["range"]=ranges
ho["time"]=times
ho["range_rate"]=range_rates
ho["snr"]=snrs
ho["angle"]=angles
ho["oid"]=oids
ho.close()

plt.hist(angles,bins=20)
plt.show()


plt.scatter(range_rates/1e3/2,ranges/1e3/2,lw=0,s=2)
plt.title("N_det %d"%(len(ranges)))
plt.xlim([-2,2])
plt.ylim([0,2000])
plt.show()
times_h=(times-n.min(times))/3600.0
plt.plot(times_h,ranges/1e3/2,".")
plt.show()

plt.hist(times_h,bins=24)
plt.show()            

plt.plot(times_h,range_rates/1e3/2,".")
plt.show()            


plt.hist(ranges/1e3/2,bins=n.linspace(0,2000,num=100))
plt.show()

plt.hist(range_rates/1e3/2,bins=100)
plt.show()


plt.hist(10.0*n.log10(snrs),bins=50)
plt.show()

