import sys

import numpy as np
import matplotlib.pyplot as plt

beam_width = 1.26 #deg
max_sat_speed = 8e3 #m/s

els = ['90.0', '88.73987398739877', '87.4357183693167', '86.1720061939742', '84.90239189256025', '83.628529014365', '82.35204237515919', '81.07452478944677', '79.77710211752762', '78.47281161299796', '77.16391480976668', '75.83375939658995', '74.49527977424731', '73.13358804398229', '71.753175185289', '70.3584796645483', '68.93771910430019', '67.48130276816576', '65.99689473774336', '64.48487395668005', '62.93289943117691', '61.33878768753357', '59.69647098103616', '57.99766296011879', '56.23923388331565', '54.40755219009088', '52.44010673902446', '50.4247836215483', '48.29222295539205', '46.01278403125778', '43.55617730018973', '40.8717857555977', '37.886094818016815', '34.478961143266076', '30.0']

IPP = 10e-3
n_IPP = 10

def calc_fence_params(leak_proof_at = 350e3, min_el_ind = len(els)):

    _els = els[:min_el_ind]
    #print('MIN_EL = {:.4f}'.format(float(_els[-1])))
    _els = [float(el) for el in _els]
    _els = _els[1:][::-1] + _els
    
    beam_arc_len = np.radians(beam_width)*leak_proof_at
    sat_traverse_time = beam_arc_len/max_sat_speed #also max scan time
    
    rn = leak_proof_at/np.sin(np.radians(_els))
    
    dwell_time = sat_traverse_time/float(len(_els))
    
    ipp_fence = np.round(dwell_time/IPP)
    if ipp_fence > n_IPP:
        ipp_fence = float(n_IPP)
    
    sens = ipp_fence/float(n_IPP)
    
    _sens = ipp_fence
    
    _els = np.array(_els)
    
    #account for pointing loss
    sens = sens*(np.sin(np.radians(_els))**2)
    sens *= (leak_proof_at/rn)**4
    
    ret = dict(
        els = _els,
        min_el = _els[0],
        dwell_time = dwell_time,
        scan_time = dwell_time*len(_els),
        sensitivity_loss = sens,
        _sens = _sens,
    )
    
    return ret


#plot1
#part_elis = []
#for ind in range(4):
#    elis = list(range(2+ind*2, len(els)-1, 8))
#    part_elis.append(elis)

#elis = [6, 7, 8, 9, 12, 13, 14, 15, 16]
elis = list(range(2, len(els), 1))

_alt = 400e3

#fig, axs = plt.subplots(2, 2, figsize=(15,7), sharex=True, sharey=True)
#axs = axs.flatten()

fig, ax = plt.subplots(figsize=(15,7))

for ind, eli in enumerate(elis):
    new_ret = calc_fence_params(leak_proof_at = _alt, min_el_ind = eli)
    
    if ind > 0:
        if int(new_ret['_sens']) != int(ret['_sens']) or ind == len(elis)-1:
            print(new_ret['_sens'], ret['_sens'])
            if ind == len(elis)-1:
                ret = new_ret
            maxi = np.argmax(ret['els'])
            ax.plot(ret['els'][:maxi], 10.0*np.log10(ret['sensitivity_loss'][:maxi]), label='Min elevation {:.2f} deg'.format(ret['els'][0]), alpha=0.75)

    ret = new_ret

ax.set_xlabel('Pointing elevation [deg]', fontsize=20)
ax.set_ylabel('Signal strength loss [dB]', fontsize=20)


#fig.text(0.5, 0.02, 'Pointing elevation [deg]', ha='center',fontsize=20)
#fig.text(0.04, 0.5, 'Signal strength loss [dB]', va='center', rotation='vertical',fontsize=20)


ax.legend()

fig.suptitle('Leak proof fence at {:.1f} km altitude with {:.2f} s coherent integration in zenith as nominal'.format(_alt*1e-3, IPP*n_IPP),fontsize=17)

plt.show()