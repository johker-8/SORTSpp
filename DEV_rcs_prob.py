import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import antenna_library as alib
import radar_library as rlib
import coord
import scipy


def draw_sph(num, min_el):
    num_done = 0
    ret = n.empty((3,num), dtype=n.float)
    while num_done < num:
        k0 = n.random.rand(4)*2.0 - 1.0
        k0n = k0.dot(k0)
        if k0n < 1.0:
            kz = (k0[0]**2 + k0[3]**2 - k0[1]**2 - k0[2]**2)/k0n
            
            if n.arctan(kz) >= n.radians(min_el):
                ret[0, num_done] = 2.0*(k0[1]*k0[3] + k0[0]*k0[2])/k0n
                ret[1, num_done] = 2.0*(k0[2]*k0[3] - k0[0]*k0[1])/k0n
                ret[2, num_done] = kz
                num_done += 1
    
    return ret


def rcs_psf(radar, txi, rxi, az, el, SNR, R, samp = 1000, min_el = 30.0, RCS_lim = 10.0, hist_res = 25):
    tx = radar._tx[txi]
    rx = radar._rx[rxi]
    
    G = tx.beam
    G.point(az, el)
    
    num = 0
    it = 0
    k_set = n.empty((3,), dtype=n.float)
    k_all = n.empty((3,samp), dtype=n.float)
    sig = n.empty((samp,), dtype=n.float)
    
    while num < samp:
        it += 1
        
        k_set = draw_sph(1, min_el)
        k_set.shape = (k_set.size,)
        
        k_ecef = coord.azel_ecef(G.lat, G.lon, 0.0, n.degrees(n.arctan2(k_set[1],k_set[0])), n.degrees(n.arcsin(k_set[2])))
        Gc = G.gain(k_ecef)
        rcs = SNR*(4.0*n.pi)**3*R**4*scipy.constants.k*rx.rx_noise*tx.coh_int_bandwidth/(tx.tx_power*Gc**2*tx.wavelength)
        
        if rcs < RCS_lim:
            print('iter {}, samp {}, rcs {} m^2 dB'.format(it,num,10.0*n.log10(rcs)))
            
            sig[num] = rcs
            k_all[0,num] = k_set[0]
            k_all[1,num] = k_set[1]
            k_all[2,num] = k_set[2]
            num += 1
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.hist(n.log10(sig), hist_res, density=True)
    
    n.savetxt('sig_data.txt',sig)
    
    ax.set_xlabel('$\log_{10}(RCS)$ [1]',fontsize=24)
    ax.set_ylabel('Probability',fontsize=24)
    ax.set_title('RCS distribution: SNR {} dB, R {} km, (az, el) = ({}, {}), samp include {} %'.format(
        10.0*n.log10(SNR),
        R*1e-3,
        az,
        el,
        float(it)/float(num)*100.0
    ), fontsize=20)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(k_all[0,:], k_all[1,:], k_all[2,:], '.b')
    ax.set_xlabel('$k_x$ [1]',fontsize=24)
    ax.set_ylabel('$k_y$ [1]',fontsize=24)
    
    ax.set_aspect('equal')
    max_range = 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    plt.show()
    
    return sig, k_all


if __name__ == '__main__':
    #initialize the radar setup
    radar = rlib.eiscat_3d()
    
    radar.set_FOV(max_on_axis=25.0,horizon_elevation=30.0)
    radar.set_SNR_limits(min_total_SNRdb=10.0,min_pair_SNRdb=0.0)
    radar.set_TX_bandwith(bw = 1.0e6)
    radar.set_beam('TX', alib.e3d_array_beam_stage1(opt='dense') )
    radar.set_beam('RX', alib.e3d_array_beam() )
    
    rcs, k = rcs_psf(
        radar,
        txi = 0,
        rxi = 0,
        az = 0.0,
        el = 90.0,
        SNR = 15.0,
        R = 400e3,
        samp = 200,
        min_el = 30.0,
        RCS_lim = 10.0,
        hist_res = int(n.sqrt(200)),
    )
