import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

'''A reward function takes in the current time and a row of the "track" array and calculates a reward-metric of observation based on the configuration.

The row format is a numpy structured array:

.. code-block:: python

    dtype = [
        ('t0', self._type),
        ('dt', self._type),
        ('tracklet', np.bool),
        ('tracklet_index', np.int64),
        ('tracklet_len', np.int64),
        ('SNRdB', self._type),
        ('SNRdB-t', self._type),
        ('index', np.int64),
        ('baselines', np.int64),
        ('type', '|S8'),
    ]

It also takes keyword arguments to allow for future expansion.

'''

def rewardf_TEMPLATE(t, track, config, **kw):
    pass


def rewardf_exp_peak_SNR_tracklet_len(t, track, config, **kw):
    '''Reward function that uses time from peak SNR and the number of accumulated data points as parameters to a normal and exponential distribution.

    **Config:**

        * sigma_t [float]: Desc
        * lambda_N [float]: Desc
    '''

    dt = t - track['SNRdB-t']
    N = track['tracklet_len']
    return np.exp(-0.5*((np.abs(dt)/config['sigma_t'])**2)) * np.exp(-N/config['lambda_N'])


def rewardf_exp_peak_SNR(t, track, config, **kw):
    '''Desc
    '''
    dt = t - track['SNRdB-t']
    return np.exp(-0.5*((np.abs(dt)/config['sigma_t'])**2))


'''
def que_value_dyn_v2(t, track, config):
    peak_snr = data['peak_snr']
    tracklets_n = data['tracklets']
    dt = data['dt']
    N = data['N']

    fN = lambda N: n.exp(-N/config['N_rate'])*config['N_scale']
    fSNR = lambda dt: n.exp(-0.5*( ((dt-config['dt_offset'])/((1.0 + float(dt-config['dt_offset'] >= 0)*config['dt_sqew'])*config['dt_sigma']))**2))*config['dt_scale']
    ftracklets = lambda tracklets: n.exp(-tracklets/config['tracklets_rate'])*config['tracklets_scale']
    fpeak = lambda peak_snr: n.exp(-peak_snr/config['peak_snr_rate'])*config['peak_snr_scale']
    ftracklet_complete = lambda N: 1.0 + (float(N)/config['tracklet_completion_rate'] - 1.0)*float(N <= config['tracklet_completion_rate'])

    if data['source'] == 'track':
        source_f = config['track-scan_ratio']
    elif data['source'] == 'scan':
        source_f = 1.0
    else:
        source_f = 1.0

    tracklets = n.sum( ftracklet_complete(N) for N in tracklets_n )

    #if n.sum(tracklets_n) == 0 and dt < 3*60.: #if we have a tracklet with no points, make sure it gets to top of que
    #    return 1e6


    if debug:
        fig, axs = plt.subplots(2, 2, figsize=(14, 10), tight_layout=True)
        x = n.linspace(1.,100,num=1000)
        axs[0,0].plot(x,fpeak(x))
        axs[0,0].set(ylabel='weight',xlabel='peak_snr [dB]')
        x = n.arange(1,500)
        axs[1,0].plot(x,fN(x))
        axs[1,0].set(ylabel='weight',xlabel='measurnment points')
        x = n.linspace(-15.0*60,15.0*60.0,num=1000)
        fsnrv = n.empty(x.shape)
        for i in range(1000):
            fsnrv[i] = fSNR(x[i])
        axs[0,1].plot(x/60.0,fsnrv)
        axs[0,1].set(xlabel='time from peak SNR [min]')
        x = n.arange(1,10)
        axs[1,1].plot(x,ftracklets(x))
        axs[1,1].set(xlabel='number of passes')
        fig, axs = plt.subplots(1, 1, figsize=(14, 10), sharey=True, tight_layout=True)
        x = n.arange(1,51)
        ftr = n.empty(x.shape)
        for i in range(50):
            ftr[i] = ftracklet_complete(x[i])
        axs.plot(x,ftr)
        axs.set(
            xlabel='Number of tracklet points', 
            ylabel='Tracklet completion', 
            title='Tracklet completion function')
        plt.show()

    ret_val = 1.0
    if config['N_on']:
        ret_val*=fN(N)
    if config['dt_on']:
        ret_val*=fSNR(dt)
    if config['tracklets_on']:
        ret_val*=ftracklets(tracklets)
    if config['peak_snr_on']:
        ret_val*=fpeak(peak_snr)
    if config['source_on']:
        ret_val*=source_f
    
    return ret_val

'''