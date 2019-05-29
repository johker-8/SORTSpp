#!/usr/bin/env python
'''Radar signal to noise calculations for hard targets.

Hard target radar echo signal to noise calculations
Range and range-rate error analysis
'''

import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
import scipy.stats as stats
import scipy.special as special
import scipy.signal as ss
import scipy.interpolate as si
import h5py
import os

# do we pop-up plots for range and range-rate error sweeps
show_plots=False

def hard_target_enr(gain_tx, gain_rx,
                    wavelength_m, power_tx,
                    range_tx_m, range_rx_m,
                    diameter_m=0.01, bandwidth=10,
                    rx_noise_temp=150.0):
    '''
    Deterine the energy-to-noise ratio for a hard target (signal-to-noise ratio).
    Assume a smooth transition between Rayleigh and optical scattering.

    gain_tx - transmit antenna gain, linear
    gain_rx - receiver antenna gain, linear
    wavelength_m - radar wavelength (meters)
    power_tx - transmit power (W)
    range_tx_m - range from transmitter to target (meters)
    range_rx_m - range from target to receiver (meters)
    diameter_m - object diameter (meters)
    bandwidth - effective receiver noise bandwidth for incoherent integration (tx_len*n_ipp/sample_rate)
    rx_noise_temp - receiver noise temperature (K)
    (Markkanen et.al., 1999)
    '''
    ##
    ## Determine returned signal power, given diameter of sphere
    ## Ignore Mie regime and use either optical or rayleigh scatter
    ##
    is_rayleigh = diameter_m < wavelength_m/(n.pi*n.sqrt(3.0))
    is_optical = diameter_m >= wavelength_m/(n.pi*n.sqrt(3.0))
    rayleigh_power = (9.0*power_tx*(((gain_tx*gain_rx)*(n.pi**2.0)*(diameter_m**6.0))/(256.0*(wavelength_m**2.0)*(range_rx_m**2.0*range_tx_m**2.0))))
    optical_power = (power_tx*(((gain_tx*gain_rx)*(wavelength_m**2.0)*(diameter_m**2.0)))/(256.0*(n.pi**2)*(range_rx_m**2.0*range_tx_m**2.0)))
    rx_noise = c.k*rx_noise_temp*bandwidth
    return(((is_rayleigh)*rayleigh_power + (is_optical)*optical_power)/rx_noise)


def target_diameter(gain_tx, gain_rx,
                    wavelength_m, power_tx,
                    range_tx_m, range_rx_m,
                    enr=1.0, bandwidth=10.0,
                    rx_noise_temp=150.0):
    '''
     Given SNR, determine the diameter of the target
    
     determine smallest sphere detactable with a certain ENR
     Ignore Mie regime and use either optical or Rayleigh scatter.
    (Markkanen et.al., 1999)
    '''
    n_diam=1000
    diams = 10**n.linspace(-4,3,num=n_diam)
    diams[n_diam-1]=n.nan
    enrs = hard_target_enr(gain_tx,gain_rx,wavelength_m,power_tx,range_tx_m,range_rx_m,diams,bandwidth,rx_noise_temp)
    i=0
    while enrs[i] < enr and i<n_diam:
        i+=1
    return(diams[i])

def simulate_echo(codes,t_vecs,bw=1e6,dop_Hz=0.0,range_m=1e3,plot=False,sr=5e6):
    '''
    Simulate a radar echo with range and doppler.
    Use windowing to simulate a continuous finite bandwidth signal.
    This is used for linearized error estimates of range and range-rate errors.
    '''

    codelen=len(codes[0])
    n_codes=len(codes)
    tvec=n.zeros(codelen*n_codes)
    z=n.zeros(codelen*n_codes,dtype=n.complex64)
    for ci,code in enumerate(codes):
        z[n.arange(codelen)+ci*codelen]=code
        tvec[n.arange(codelen)+ci*codelen]=t_vecs[ci]
    tvec_i=n.copy(tvec)
    tvec_i[0]=tvec[0]-1e99
    tvec_i[len(tvec)-1]=tvec[len(tvec)-1]+1e99        
    zfun=si.interp1d(tvec_i,z,kind="linear")
    dt = 2.0*range_m/c.c

    z=zfun(tvec+dt)*n.exp(1j*n.pi*2.0*dop_Hz*tvec)
    if plot:
        plt.plot(tvec,z.real)
        plt.show()
    return(z)

def lin_error(enr=10.0,txlen=1000.0,n_ipp=10,ipp=20e-3,bw=1e6,dr=10.0,ddop=1.0,sr=100e6,plot=False):
    '''
     Determine linearized errors for range and range-rate error
     for a psuedorandom binary phase coded radar transmit pulse
     with a certain transmit bandwidth (inverse of bit length)

     calculate line of sight range and range-rate error,
     given ENR after coherent integration (pulse compression)
     txlen in microseconds.

    Simulate a measurement and do a linearized error estimate.

    '''
    codes=[]
    t_vecs=[]    
    n_bits = int(bw*txlen/1e6)
    oversample=int(sr/bw)
    wfun=ss.hamming(oversample)
    wfun=wfun/n.sum(wfun)
    for i in range(n_ipp):
        bits=n.array(n.sign(n.random.randn(n_bits)),dtype=n.complex64)
        zcode=n.zeros(n_bits*oversample+2*oversample,dtype=n.complex64)
        for j in range(oversample):
            zcode[n.arange(n_bits)*oversample+j+oversample]=bits
        # filter signal so that phase transitions are not too sharp
        zcode=n.convolve(wfun,zcode,mode="same")
        codes.append(zcode)
        tcode=n.arange(n_bits*oversample+2*oversample)/sr + float(i)*ipp
        t_vecs.append(tcode)

    z0=simulate_echo(codes,t_vecs,dop_Hz=0.0,range_m=0.0,bw=bw,sr=sr)
    tau=(float(n_ipp)*txlen/1e6)
    
    # convert coherently integrated ENR to SNR (variance of the measurement errors at receiver bandwidth)
    snr = enr/(tau*sr)

    if plot:
        plt.plot(z0.real)
        plt.plot(z0.imag)
        plt.title("Echo")
        plt.show()

    z_dr=simulate_echo(codes,t_vecs,dop_Hz=0.0,range_m=dr,bw=bw,sr=sr)
    z_diff_r=(z0-z_dr)/dr    
    if plot:
        plt.plot(z_diff_r.real)
        plt.plot(z_diff_r.imag)
        plt.title("Range derivative of echo (dm/dr)")
        plt.show()
    z_ddop=simulate_echo(codes,t_vecs,dop_Hz=ddop,range_m=0.0,bw=bw,sr=sr)
    z_diff_dop=(z0-z_ddop)/ddop
    
    if plot:
        plt.plot(z_diff_dop.real)
        plt.plot(z_diff_dop.imag)
        plt.title("Doppler derivative of echo (dm/df)")        
        plt.show()

    t_l=len(z_dr)
    A=n.zeros([t_l,2],dtype=n.complex64)
    A[:,0]=z_diff_r
    A[:,1]=z_diff_dop
    S=n.real(n.linalg.inv(n.dot(n.transpose(n.conj(A)),A))/snr)

    return(n.sqrt(n.diag(S)))

# based on monte-carlo simulations
# returns 1-sigma error in meters
def ionospheric_error_fun():
    r=n.array([0.0,1000.0,1000000.0])
    e=n.array([0.0,0.1,100.0])*150.0
    efun=si.interp1d(r,e)
    return(efun)

# doppler error
def precalculate_dr(txlen,bw,ipp=20e-3,n_ipp=20,n_interp=20):
    if os.path.exists("cache_error.h5"):
        h=h5py.File("cache_error.h5","r")
        enrs=n.copy(h["enrs"].value)
        drs=n.copy(h["drs"].value)
        ddops=n.copy(h["ddops"].value)
        h.close()
    else:
        enrs=10.0**n.linspace(-3,10,num=n_interp)
        drs=n.zeros(n_interp)
        ddops=n.zeros(n_interp)    
        for ei,s in enumerate(enrs):
            dr,ddop=lin_error(enr=s,txlen=txlen,bw=bw,ipp=ipp,n_ipp=n_ipp)
            print("enr %1.2f dr %1.2f ddop %1.2f"%(s,dr,ddop))
            drs[ei]=dr
            ddops[ei]=ddop        

        
    rfun=si.interp1d(n.log10(enrs),n.log10(drs))
    dopfun=si.interp1d(n.log10(enrs),n.log10(ddops))

    if not os.path.exists("cache_error.h5"):
        ho=h5py.File("cache_error.h5","w")
        ho["enrs"]=enrs
        ho["drs"]=drs
        ho["ddops"]=ddops        
        ho.close()
    return(rfun,dopfun)


def debug_debris_filter():
    '''
    Debug plot 

    Simulate the effect of the planned strong SNR satellite filter for EISCAT 3D.
    '''
    n_r=1000
    ranges = n.linspace(300e3,2000e3,num=n_r)
    filter_diams=n.zeros(n_r)
    diams=n.zeros(n_r)    
    for ri,r in enumerate(ranges):
        filter_diams[ri] = target_diameter(10**4.3, 10**2.263, 230e6/c.c, 5e6, ranges[ri], ranges[ri], enr=100.0, bandwidth=500.0, rx_noise_temp=150.0)
        diams[ri] = target_diameter(10**4.3, 10**4.3, 230e6/c.c, 5e6, ranges[ri], ranges[ri], enr=25.0, bandwidth=45.0, rx_noise_temp=150.0)
    plt.loglog(ranges/1e3,filter_diams,label="Filtered objects")
    plt.loglog(ranges/1e3,diams,label="Detectable objects")
    plt.xlabel("Range (km)")
    plt.ylabel("Target diameter (m)")
    plt.title("EISCAT 3D hard target filter")
    plt.grid()
    plt.legend()
    plt.savefig("report_plots/sat_filter.pdf")
    if show_plots:
        plt.show()
    else:
        plt.clf()
        plt.close()        

def debug_tx_len_sweep(tlen=n.linspace(10.0,2000.0,num=10),bw=1e6,enr=1000.0,n_ipp=10,ipp=20e-3):
    '''
    Debug plot

    Sweep through different transmit pulse lengths and determine
    the measurement variance of range and range-rate estimates.
    Consider all echoes to have constant SNR
    '''

    drs=[]
    ddops=[]    
    for tl in tlen:
        dr,ddop=lin_error(enr=enr,txlen=tl,bw=1e6,n_ipp=n_ipp,ipp=ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)        
    plt.semilogy(tlen,drs)
    plt.ylim([n.min(drs)/15.0,n.max(drs)*15.0])    
    plt.title("Fixed ENR=%1.0f n_ipp=%d IPP=%1.2f (ms)"%(enr,n_ipp,ipp*1e3))
    plt.xlabel("TX pulse length ($\mu$s)")
    plt.ylabel("Range error standard deviation (m)")
    plt.grid()        
    plt.subplot(122)
    plt.semilogy(tlen,ddops)
    plt.title("Fixed ENR=%1.0f"%(enr))
    plt.xlabel("TX pulse length ($\mu$s)")
    plt.ylabel("Doppler error standard deviation (Hz)")
    plt.ylim([n.min(ddops)/15.0,n.max(ddops)*15.0])
    plt.grid()
    plt.tight_layout()
    plt.savefig("report_plots/deb_tx_len_sweep_const_enr.pdf")
    if show_plots:
        plt.show()
    else:
        plt.clf()
        plt.close()        



def debug_ipp_sweep(n_ipps=n.arange(1,20),bw=1e6,enr=1000.0,ipp=20e-3,txlen=2000.0):
    '''
    Debug IPP sweep plot.

    Determine the measurement variance of range and range-rate estimates
    for a different number of transmit pulses that are coherently integrated.
    
    '''
    
    drs=[]
    ddops=[]    
    for n_ipp in n_ipps:
        print(n_ipp)
        ratio=float(n_ipp)/float(n.max(n_ipps))
        print(ratio)        
        dr,ddop=lin_error(enr=ratio*enr,txlen=txlen,bw=1e6,n_ipp=n_ipp,ipp=ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)        
    plt.loglog(n_ipps,drs)
    plt.title("Maximum ENR=%1.0f IPP=%1.2f (ms)"%(enr,ipp*1e3))
    plt.xlabel("Number of IPPs to integrate")
    plt.ylabel("Range error standard deviation (m)")
    plt.grid()        
    plt.subplot(122)
    plt.loglog(n_ipps,ddops)
    plt.xlabel("Number of IPPs to integrate")
    plt.ylabel("Doppler error standard deviation (Hz)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("report_plots/deb_n_ipp_sweep.pdf")
    if show_plots:
        plt.show()
    else:
        plt.clf()
        plt.close()        
    
def debug_tx_len_sweep2(tlen=n.linspace(10.0,2000.0,num=10),bw=1e6,enr0=1000.0,n_ipp=20):
    '''
    Debug plot

    Determine variance for different transmit pulse lengths with fixed
    peak power (less SNR for shorter pulses)
    '''

    drs=[]
    ddops=[]
    maxtlen=n.max(tlen)
    for tl in tlen:
        dr,ddop=lin_error(enr=(tl/maxtlen)*enr0,txlen=tl,bw=bw,n_ipp=n_ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)
    plt.loglog(tlen,drs)
    plt.title("Fixed $P_{\mathrm{tx}}$, Max ENR=%1.0f"%(enr0))
    plt.xlabel("TX pulse length ($\mu$s)")
    plt.ylabel("Range error standard deviation (m)")
    plt.grid()        
    plt.subplot(122)
    plt.loglog(tlen,ddops)
    plt.title("Fixed $P_{\mathrm{tx}}$, Max ENR=%1.0f"%(enr0))
    plt.xlabel("TX pulse length ($\mu$s)")
    plt.ylabel("Doppler error standard deviation (Hz)")
    plt.grid()    
    plt.tight_layout()
    plt.savefig("report_plots/deb_tx_len_sweep_const_txp.pdf")
    if show_plots:
        plt.show()
    else:
        plt.clf()
        plt.close()        


#
# Determine the variance of range and range-rate estimation error for transmit
# pulses with different transmit bandwidth.
#    
def debug_bw_sweep(bw=n.linspace(0.01e6,5e6,num=10),txlen=1000.0,enr=100.0,n_ipp=20,ipp=20e-3):
    drs=[]
    ddops=[]    
    for b in bw:
        print("bw %1.2f"%(b))
        dr,ddop=lin_error(enr=enr,txlen=txlen,bw=b,n_ipp=n_ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)
    plt.loglog(bw/1e6,drs)
    plt.xlabel("TX bandwidth (MHz)")
    plt.ylabel("Range error standard deviation (m)")
    plt.title("ENR=%1.0f txlen=%1.2f (ms)\nn_ipp=%d ipp=%1.2f (ms)"%(enr,txlen/1e3,n_ipp,ipp*1e3))
    plt.grid()        
    plt.subplot(122)    
    plt.semilogy(bw/1e6,ddops)
    plt.ylim([n.min(ddops)/10.0,n.max(ddops)*10.0])
    plt.title("Fixed ENR=%1.0f"%(enr))    
    plt.xlabel("TX bandwidth (MHz)")
    plt.ylabel("Doppler error standard deviation (Hz)")
    plt.grid()    
    plt.tight_layout()
    plt.savefig("report_plots/deb_bw_sweep_lin.pdf")
    if show_plots:
        plt.show()
    else:
        plt.clf()
        plt.close()        
    

def debug_enr_sweep(enrs=10.0**n.linspace(0,6.0,num=10),txlen=1000.0,bw=1e6,n_ipp=10,ipp=20e-3):
    '''
    Debug plot

    Sweep through different energy-to-noise ratios and determine range and Doppler error variance.
    '''
    drs=[]
    ddops=[]    
    for s in enrs:
        dr,ddop=lin_error(enr=s,txlen=txlen,bw=bw,n_ipp=n_ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)
    plt.semilogy(10.0*n.log10(enrs),drs)
    plt.title("TX len %1.0f $\mu$s IPP=%1.2f (ms)\nn_ipp=%d TX BW %1.0f MHz"%(txlen,ipp*1e3,n_ipp,bw/1e6))
    plt.xlabel("ENR (dB)")
    plt.ylabel("Range error standard deviation (m)")
    plt.grid()        
    plt.subplot(122)    
    plt.semilogy(10.0*n.log10(enrs),ddops)
    plt.xlabel("ENR (dB)")
    plt.ylabel("Doppler error standard deviation (Hz)")
    plt.grid()    
    plt.tight_layout()
    plt.savefig("report_plots/deb_enr_sweep_lin.pdf")
    if show_plots:
        plt.show()
    else:
        plt.clf()
        plt.close()        
    

def debug_rcs_sweep():
    ''' 
    Debug plot.

    Sweep through different diameters and calculate radar cross-section. 
    The aim is to validate the perfectly conducting sphere model. Plot result.
    '''
    diams=10.0**n.linspace(-3,3,num=200)
    snrs=hard_target_enr(gain_tx=10**4.8,
                         gain_rx=10**4.8,
                         wavelength_m=0.32,
                         power_tx=1.6e6,
                         range_tx_m=1000e3,
                         range_rx_m=1000e3,
                         diameter_m=diams,
                         bandwidth=45.0,
                         rx_noise_temp=100.0)

    plt.axvline(0.32/(n.pi*n.sqrt(3.0)),color="red",alpha=0.2)
    plt.axhline(25.0,color="green",alpha=0.2)    

    plt.text(0.32/(n.pi*n.sqrt(3.0)),1e7,"$d=\\frac{\lambda}{\pi \sqrt{3.0}}$")
    plt.text(1e-3,50,"$\mathrm{SNR}=25$")
    plt.loglog(diams,snrs)
    
    plt.xlabel("Diameter (m)")
    plt.ylabel("Signal-to-noise ratio (linear)")
    plt.title("Perfect conducting sphere scattering model\n($\lambda=0.32$ m, G=48 dB, R=1000 km, B=45  Hz, T=100 K, P=1.6 MW)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("report_plots/snr_plot.pdf")
    if show_plots:
        plt.show()
    else:
        plt.clf()
        plt.close()        
    
if __name__ == "__main__":
    
    debug_bw_sweep()    
    debug_debris_filter()    
    debug_rcs_sweep()
    
    debug_ipp_sweep()    
    debug_enr_sweep()
    debug_tx_len_sweep()

    debug_tx_len_sweep2()


