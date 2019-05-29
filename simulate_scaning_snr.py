#!/usr/bin/env python

'''Functions for single object propagation and SNR examination.

'''

import numpy as n
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt
#scaning snr curve

#SORTS Libraries
import radar_library as rl
import radar_scan_library as rslib
import scheduler_library as sch
import antenna_library as alib
import simulate_tracking as st
import space_object as so
from antenna import full_gain2inst_gain,inst_gain2full_gain

def simulate_full_scaning_snr_curve(radar,o,det_times,tresh,rem_t,obs_par,groups,IPP_scale=1.0, plot = True,verbose=True):

    tresh = full_gain2inst_gain(gain = tresh, \
        groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale)

    if verbose:
        print('Running sweep with limits: %.2f dB to %.2f dB in subgroup gain'%(tresh,rem_t))
        print('Running sweep with limits: %.2f dB to %.2f dB in integrated gain'%(\
            inst_gain2full_gain(tresh,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale), \
            inst_gain2full_gain(rem_t,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale)))
    for txi,dets in enumerate(det_times['t']):
        full_snr_curve = []
        full_t_curve = []
        for I in range(len(radar._rx)):
            full_t_curve.append([])
            full_snr_curve.append([])
        for det in dets:
            t1,t2 = det
            
            tnum = (t2-t1)/obs_par.coher_int_t
            t = n.linspace(t1,t2,num=int(tnum*2))

            all_snrs = st.get_scan_snr(t,o,radar)
            all_snrs = 10.0*n.log10(n.array(all_snrs))
            all_snrs_array = all_snrs.copy()
            all_snrs = full_gain2inst_gain(gain = all_snrs, \
                groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale)
            snrs = all_snrs[txi]

            max_snr = tresh
            for snr in snrs:
                if n.max(snr) > max_snr:
                    max_snr = n.max(snr)

            if verbose:
                print('Calculating SNR between %.4f h and %.4f h: max = %.2f dB'%(t1/3600.,t2/3600.,max_snr))
            
            if max_snr > tresh:
                if plot:
                    fig, axs = plt.subplots(len(snrs), 1, figsize=(14, 10), sharex=True, tight_layout=True)

                for I,snr in enumerate(snrs):
                    idx = snr > tresh
                    tv = t[idx]/3600.0
                    if len(tv) > 0:
                        snrv = snr[idx]
                        full_t_curve[I] += tv.tolist()
                        full_snr_curve[I] += snrv.tolist()
                        if plot:
                            axs[I].plot(tv,snrv,'.b')
                            axs[I].plot([tv[0],tv[-1]] ,[rem_t,rem_t],'-r')
                            axs[I].plot([tv[0],tv[-1]] ,[tresh,tresh],'-g')
                            axs[I].set( \
                                title='Subgroup instantaneous SNR: %s to %s'%(radar._tx[txi].name,radar._rx[I].name), \
                                ylabel='SNR [dB]', \
                                xlabel='time [h]')
        if plot:
            fig, axs = plt.subplots(len(full_t_curve), 1, figsize=(14, 10), tight_layout=True)
            for I,vals in enumerate(zip(full_t_curve,full_snr_curve)):
                tv,snrc = vals
                if len(snrc) > 0:
                    axs[I].plot(n.array(tv),\
                        inst_gain2full_gain(n.array(snrc),groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale),'.b')
                    axs[I].plot([tv[0],tv[-1]] ,[\
                        inst_gain2full_gain(rem_t,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale),\
                        inst_gain2full_gain(rem_t,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale)],'-r')
                    axs[I].plot([tv[0],tv[-1]] ,[\
                        inst_gain2full_gain(tresh,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale),\
                        inst_gain2full_gain(tresh,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale)],'-g')
                    axs[I].set( \
                        title='Coherently integrated SNR: %s to %s'%(radar._tx[txi].name,radar._rx[I].name), \
                        ylabel='SNR [dB]', \
                        xlabel='time [h]')

            fig, axs = plt.subplots(len(full_t_curve), 1, figsize=(14, 10), tight_layout=True)
            for I,vals in enumerate(zip(full_t_curve,full_snr_curve)):
                tv,snrc = vals
                if len(snrc) > 0:
                    axs[I].plot(n.arange(len(tv)),\
                        inst_gain2full_gain(n.array(snrc),groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale),'.b')
                    axs[I].plot([0,len(tv)-1] ,[\
                        inst_gain2full_gain(rem_t,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale),\
                        inst_gain2full_gain(rem_t,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale)],'-r')
                    axs[I].plot([0,len(tv)-1] ,[\
                        inst_gain2full_gain(tresh,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale),\
                        inst_gain2full_gain(tresh,groups = groups, N_IPP = obs_par.N_IPP, IPP_scale=IPP_scale)],'-g')
                    axs[I].set( \
                        title='Coherently integrated SNR: %s to %s'%(radar._tx[txi].name,radar._rx[I].name), \
                        ylabel='SNR [dB]', \
                        xlabel='index [1]')
    if plot:
        plt.show()

    return(full_t_curve,full_snr_curve )