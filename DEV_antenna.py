#!/usr/bin/env python
#
#
import numpy as n
import scipy.constants as c
import scipy.special as s
import matplotlib.pyplot as plt

import coord
import antenna_library as alib
import antenna

def test_planar():
    for phased_el in n.linspace(3,90,num=10):
        print(phased_el)
        bp=alib.planar_beam(0.0,phased_el,60,10,az1=0.0,el1=90.0,a0=40.0,I_0=10**4.3,f=230e6)
 #       print(bp.I_1)
        gains=[]
        els=n.linspace(0.0,90.0,num=1000)
        for ei,e in enumerate(els):
            k=coord.azel_ecef(60.0, 10.0, 0.0, 0.0, e)
#            print("phased el %f k el %f"%(phased_el,e))
            g=bp.gain(k)
            gains.append(g)
        gains=n.array(gains)
        plt.plot(els,10.0*n.log10(gains),label="el=%1.2f"%(phased_el))
        
        plt.ylim([0,50])
        plt.axvline(phased_el,color="black")
    plt.legend()
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("Gain (dB)")
    plt.title("Planar array gain as a function of pointing direction")
    plt.show()

def test_planar4():
    bp=alib.planar_beam(0.0,45.0,60,10,az1=0.0,el1=90.0,a0=40.0,I_0=10**4.3,f=230e6)    
    for phased_el in n.linspace(3,90,num=10):
        bp.point(0.0,phased_el)
        gains=[]
        els=n.linspace(0.0,90.0,num=1000)
        for ei,e in enumerate(els):
            k=coord.azel_ecef(60.0, 10.0, 0.0, 0.0, e)
            g=bp.gain(k)
            gains.append(g)
        gains=n.array(gains)
        plt.plot(els,10.0*n.log10(gains),label="el=%1.2f"%(phased_el))
        
        plt.ylim([0,50])
        plt.axvline(phased_el,color="black")
    plt.legend()
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("Gain (dB)")
    plt.title("Planar array gain as a function of pointing direction")
    plt.show()
    
def test_planar2():
    el_phase=30.0
    az_phase=40.0    
    B=n.zeros([500,500])
    els=n.linspace(0,90,num=500)
    azs=n.linspace(0,360,num=500)
    bp=alib.planar_beam(az_phase,el_phase,60,19,az1=0.0,el1=90.0,a0=16.0,I_0=10**4.3,f=230e6)
    for ei,e in enumerate(els):
        for ai,a in enumerate(azs):
            k=coord.azel_ecef(60.0, 19.0, 0.0, a, e)
            B[ei,ai]=bp.gain(k)
    dB=10.0*n.log10(B)
    m=n.max(dB)
    plt.pcolormesh(azs,els,10.0*n.log10(B),vmin=m-20.0,vmax=m)
    plt.axhline(el_phase)
    plt.axvline(az_phase)    
    plt.colorbar()
    plt.show()
        
def test_planar3():
    S=n.zeros([100,200])
    el_phase=90.0
    bp=alib.planar_beam(0,el_phase,60,19.0,az1=0.0,el1=el_phase,a0=16.0,I_0=10**4.3,f=230e6)
    els=n.linspace(0,90,num=100)
    azs=n.linspace(0,360,num=200)
    
    for ei,e in enumerate(n.linspace(0,90,num=100)):
        for ai,a in enumerate(n.linspace(0,360,num=200)):
            k=coord.azel_ecef(60.0, 19.0, 0.0, a, e)
            S[ei,ai]=bp.gain(k)
    plt.pcolormesh(azs,els,10.0*n.log10(S),vmin=0,vmax=100)
    plt.axvline(el_phase)
    plt.colorbar()
    plt.show()




def plot_beams():
    min_el = 80.0
    bp=alib.airy_beam(90.0,90,60,19,f=930e6,I_0=10**4.3,a=16.0)
    gains=[]
    els=n.linspace(min_el,90.0,num=1000)
    for a in els:
        k=coord.azel_ecef(60.0, 19.0, 0.0, 90, a)
        gains.append(bp.gain(k))
    gains=n.array(gains)
    plt.plot(els,10.0*n.log10(gains),label="airy")

    bp=alib.cassegrain_beam(90.0,90,60,19,f=930e6,I_0=10**4.3,a0=16.0, a1=4.58)
    gains=[]
    for a in els:
        k=coord.azel_ecef(60.0, 19.0, 0.0, 90, a)
        gains.append(bp.gain(k))
    gains=n.array(gains)
    plt.plot(els,10.0*n.log10(gains),label="cassegrain")

    bp=alib.planar_beam(0,90.0,60,19,I_0=10**4.3,f=233e6,a0=40.0,az1=0,el1=90.0)
    gains=[]
    for a in els:
        k=coord.azel_ecef(60.0, 19.0, 0.0, 90, a)
        gains.append(bp.gain(k))
    gains=n.array(gains)

    plt.plot(els,10.0*n.log10(gains),label="planar")
    plt.ylim([0,50])
    
    plt.legend()
    plt.show()


def plot_e3d_antennas():
    antennas = alib.e3d_array_stage1(233e6,opt='sparse')
    
    plt.plot(antennas[:,0],antennas[:,1],'b.')
    plt.plot(40*n.cos(n.linspace(0,2*n.pi,num=100)),40*n.sin(n.linspace(0,2*n.pi,num=100)),'-k')
    plt.title('sparse SUBGROUPS %i'%(antennas.shape[0]/91))
    plt.show()
    

    antennas = alib.e3d_array_stage1(233e6,opt='dense')
    
    plt.plot(antennas[:,0],antennas[:,1],'b.')
    plt.plot(40*n.cos(n.linspace(0,2*n.pi,num=100)),40*n.sin(n.linspace(0,2*n.pi,num=100)),'-k')
    plt.title('dense SUBGROUPS %i'%(antennas.shape[0]/91))
    plt.show()

    antennas = alib.e3d_array(233e6)
    
    plt.plot(antennas[:,0],antennas[:,1],'b.')
    plt.plot(40*n.cos(n.linspace(0,2*n.pi,num=100)),40*n.sin(n.linspace(0,2*n.pi,num=100)),'-k')
    plt.title('SUBGROUPS %i'%(antennas.shape[0]/91))
    plt.show()

def plot_e3d_beam_stering():
    e3dv=[]
    e3dv.append( alib.e3d_array_beam(az0 = 0.0,el0 = 90.0) )
    e3dv.append( alib.e3d_array_beam(az0 = 0.0,el0 = 60.0) )
    e3dv.append( alib.e3d_array_beam(az0 = 0.0,el0 = 30.0) )

    antenna.plot_gains(e3dv,min_el=0)

def plot_e3d_stages():
    e3d1=alib.e3d_array_beam_stage1(opt='sparse')
    e3d2=alib.e3d_array_beam_stage1(opt='dense')
    e3d3=alib.e3d_array_beam()

    antenna.plot_gains([e3d1,e3d2,e3d3],min_el=80.0)

def plot_compare_library_beams():
    min_el = 80
    e3d=alib.e3d_array_beam(az0 = 0.0,el0 = 90.0, I_0=10**4.3)
    bp1=alib.airy_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.3,a=40.0)
    bp2=alib.cassegrain_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.3,a0=80.0, a1=80.0/16.0*2.29)
    bp3=alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.3,a0=40.0,az1=0,el1=90.0)
    antenna.plot_gains([bp1,bp2,bp3,e3d],min_el=min_el)    


def plot_compare_eiscat_beams():
    min_el = 80
    uhf0=alib.airy_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=930e6,I_0=10**4.81,a=16.0)
    uhf1=alib.cassegrain_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=930e6,I_0=10**4.81,a0=32.0, a1=4.58)
    antenna.plot_gains([uhf1],min_el=min_el,name="(UHF)")
    
def plot_compare_esr_beams():
    min_el = 80
#    uhf0=alib.airy_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=500e6,I_0=10**4.25,a=16.0)

    uhfm=alib.uhf_beam(az0=0.0, el0=90.0, lat=60, lon=19, I_0=10**4.81, f=930e6, beam_name="Tromso")
#    uhf0=alib.cassegrain_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=930e6,I_0=10**4.81,a0=32.0, a1=5.0,beam_name="Tromso")
 #   uhf1=alib.cassegrain_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=500e6,I_0=10**4.25,a0=32.0, a1=5.0,beam_name="Svalbard")
    uhfm1=alib.uhf_beam(az0=0.0, el0=90.0, lat=60, lon=19, I_0=10**4.25, f=500e6, beam_name="Svalbard")    
    antenna.plot_gains([uhfm,uhfm1],min_el=min_el)    
    

def time_compare_library_beams():
    e3d=alib.e3d_array_beam(az0 = 0.0,el0 = 90.0, I_0=10**4.3)
    bp1=alib.airy_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.3,a=40.0)
    bp2=alib.cassegrain_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.3,a0=80.0, a1=80.0/16.0*2.29)
    bp3=alib.planar_beam(az0 = 0.0,el0 = 90.0,lat = 60,lon = 19,f=233e6,I_0=10**4.3,a0=40.0,az1=0,el1=90.0)

    import time
    k=coord.azel_ecef(e3d.lat, e3d.lon, 0.0, 0, 87.0)
    test_n = 500
    t = n.zeros((test_n,4))
    for i in range(test_n):
        t0 = time.clock()
        g = e3d.gain(k)
        t[i,0] = time.clock() - t0

        t0 = time.clock()
        g = bp1.gain(k)
        t[i,1] = time.clock() - t0

        t0 = time.clock()
        g = bp2.gain(k)
        t[i,2] = time.clock() - t0

        t0 = time.clock()
        g = bp3.gain(k)
        t[i,3] = time.clock() - t0
    
    print('Exec time %s: mean %.5f s, std %.5f s'%(e3d.beam_name,n.mean(t[:,0]),n.std(t[:,0]),))
    print('Exec time %s: mean %.5f s, std %.5f s'%(bp1.beam_name,n.mean(t[:,1]),n.std(t[:,1]),))
    print('Exec time %s: mean %.5f s, std %.5f s'%(bp2.beam_name,n.mean(t[:,2]),n.std(t[:,2]),))
    print('Exec time %s: mean %.5f s, std %.5f s'%(bp3.beam_name,n.mean(t[:,3]),n.std(t[:,3]),))

    print('Exec time %s vs %s: mean %.5f, std %.5f'%(e3d.beam_name,bp3.beam_name,n.mean(t[:,0])/n.mean(t[:,3]),n.std(t[:,0])/n.std(t[:,3]),))
    

if __name__ == "__main__":
    print('entered')
    #import DEV_antenna as da
#    plot_compare_eiscat_beams()
    plot_compare_esr_beams()    
    exit(0)
#    plot_beams()
    #plot_e3d()
    #test_planar()
    #test_planar2()
    #test_planar3()
    #test_planar4()

    #plot_e3d_stages()
    #plot_e3d_beam_stering()
    #plot_e3d_antennas()
    
    #time_compare_library_beams()



    e3d=alib.e3d_array_beam()
    antenna.plot_gain3d(e3d,res=100,min_el=85)
