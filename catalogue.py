'''Catalogue class.


NOTES:
If a object is detected it automatically produces a tracklet with one point (the point of detection)
It can then get more tracklet points from the scheduler

We will have to think like this:
* we run 1 set of obsrevation confugrations for one certain time
* if a object is discovered, it can only get tracklet points that pass
* Correlation between unknown objects will be AFTERWARDS, not in read time, thus it can be "rediscovered"
* These "rediscoveries" will help imporve orbital elelemnts when it is added to the catalouge.

'''

#Standard python
import os
import shutil
import time
import datetime
import glob
import copy

#Packages
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from mpi4py import MPI

#SORTS++
from population import Population
import dpt_tools as dpt


comm = MPI.COMM_WORLD


class Catalogue:
    '''

    # TODO: Write proper documentation for this class.


    ** BELOW IS OLD DOCS **
    _detections format:
    [the object id] -> dict
    dict: "t0" initial detection
          "t1" passes below horizon
          "snr" the snr's
          "tm" time of max SNR
    each dict is a vector where one entry is one detection
    e.g _detections[obj id 4]['t0'][detection 2]
    
    _maintinence format:, None indicate fail at that slot
    [the object id] -> dict
    dict: "t" list of all above to below horizon rimes, tx lst ['t'][tx 0][pass 2][above horizon time = 0, below = 1]
          "snr" the list of max snrs of all rx tx pairs, i,e ["snr"][tx 0][pass 0][rx 1][0=SNR,1=time]
    each dict is a vector where one entry is one detection
    e.g _maintinence[obj id 4]['t'][tx 0][pass 2][0] = above horizon t
                            
    track format [track nr], is list:
     0 : t0 (scan: detection time, track: above horizon)
     1 : dt (time untill horizon)
     2 : detected/measured?
     3 : SNR dB (scan: best detection posibility, track: peak snr)
     4 : OID [not pop-id]
     5 : number of baselines, e.g. 3=tristatic
     6 : track is maintenance "track" or discovery "scan"
     7 : time of SNR dB (col 3)
    e.g _tracks[track nr 4][3] = SNR dB
     
    _discovered format:
    [object id]
    [True or false, track number]
    e.g _discovered[object id 4][1] = track of detection
    
    
    _tracklets
    format: rows = tracks
     cols: ...fnames... [one col for each name] 
    
    #known objects DO NOT NEED TO BE SCANED FOR

    '''
    def __init__(self, population, known=False):
        self.size = len(population)
        assert self.size > 0, 'Population must contain at least one object.'

        self.population = population

        self._type = self.population._default_dtype

        self._data_format = [
            'oid', #0
            'known', #1
            't0_unix', #2
            'tracks', #3
            'tracklets', #4
            'discovered', #5
            'discovery_time', #6
        ]

        mjd0s = population['mjd0']
        self._t0_unix = np.empty((self.size,), dtype=self._type)
        for ind, mjd0 in enumerate(mjd0s):
            self._t0_unix[ind] = dpt.jd_to_unix(dpt.mjd_to_jd(mjd0))

        self._discovered = np.full((self.size,), False, dtype=np.bool)
        self._discovery_track = np.empty((self.size,), dtype=np.int64)
        self._discovery_time = np.empty((self.size,), dtype=self._type)

        self._known = np.empty((self.size,), dtype=np.bool)
        if isinstance(known, bool):
            for ind in range(self.size):
                self._known[ind] = known
        else:
            for ind in range(self.size):
                self._known[ind] = known[ind]

        self._oids = population['oid']

        self._maintinence = [None]*self.size
        self._detections = [None]*self.size
        
        self._det_fields = ["t0", "t1", "snr", 'tm', "range", "range_rate", "tx_gain", "rx_gain", "on_axis_angle"]

        self._track_format = [
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
        self.tracks = np.empty((0,), dtype=self._track_format)



        self._tracking_summary_dtype = [
            ('index', np.int64),
            ('peak-SNR-time', self._type),
            ('peak-SNRdB', self._type),
            ('tracking-time', self._type),
        ]
        self._scan_summary_dtype = [
            ('index', np.int64),
            ('detection-time', self._type),
            ('SNRdB', self._type),
            ('tracking-time', self._type),
        ]
        self._tracklet_statistics_dtype = [
            ('index', np.int64),
            ('track-id', np.int64),
            ('points-deviation-mean', self._type),
            ('points-deviation-std', self._type),
            ('points-deviation-skew', self._type),
            ('points', self._type),
            ('peak-SNRdB', self._type),
            ('span', self._type),
            ('points-deviation-normalized-mean', self._type),
            ('points-deviation-normalized-std', self._type),
            ('normalized-span', self._type),
        ]
        self._tracks_summary_dtype = [
            ('index', np.int64),
            ('tracks', self._type),
            ('type', '|S8'),
            ('tracklets', self._type),
            ('tracklet-len-sum', self._type),
            ('tracklet-len-mean', self._type),
            ('tracklet-len-std', self._type),
            ('peak-SNRdB-mean', self._type),
            ('peak-SNRdB-std', self._type),
        ]
        self._prior_dtype = [
            ('date', 'datetime64[us]'),
            ('index', np.int64),
            ('x', self._type),
            ('y', self._type),
            ('z', self._type),
            ('vx', self._type),
            ('vy', self._type),
            ('vz', self._type),
            ('cov', self._type, (6,6)),
        ]
        self.priors = np.empty((0,), dtype=self._prior_dtype)

        self._tracklet_statistics = np.empty((0, ), dtype=self._tracklet_statistics_dtype)
        self._tracks_summary = np.empty((0, ), dtype=self._tracks_summary_dtype)
        self._tracking_summary = np.empty((0, ), dtype=self._tracking_summary_dtype)
        self._scan_summary = np.empty((0, ), dtype=self._scan_summary_dtype)

        self._tracklet_format = {
            'track_id': None,
            't': None,
            'index': None,
            'fnames': None,
            'is_prior': None,
        }
        self.tracklets = []


    def _calculations(self):
        self.maintinece_summary()
        self.detection_summary()
        self.track_statistics()

    def plots(self, save_folder = None):
        self._calculations()

        self.maintinece_summary_plot(save_folder = save_folder)
        self.detection_summary_plot(save_folder = save_folder)

        self.track_statistics_plot(save_folder = save_folder)

        self.maintenance_tracks_plot(save_folder = save_folder)
        self.detection_tracks_plot(save_folder = save_folder)

    def add_tracklet(self, **kwargs):
        '''Add a tracklet to the internal list.
        '''
        new_tracklet = copy.deepcopy(self._tracklet_format)
        for key, value in kwargs.items():
            new_tracklet[key] = value
        self.tracklets.append(new_tracklet)


    def add_tracks(self, num, data=None):
        '''Add more tracks to data array.
        '''
        if data is not None:
            if isinstance(data, np.ndarray):
                self.tracks = np.append(self.tracks, data, axis=0)
            else:
                more = np.empty((num,), dtype=self._track_format)
                for ind, row in enumerate(data):
                    more[ind] = tuple(row)

                self.tracks = np.append(self.tracks, more, axis=0)

        else:
            more = np.empty((num,), dtype=self._track_format)
            self.tracks = np.append(self.tracks, more, axis=0)


    def get_orbit(self, ind, t0, t1):
        '''Get the orbit-determination for object considering tracks between certain times.
        '''
        pass


    def compile_tracks(self, radar, t0, t1, radar_control = None):
        '''Takes a radar system and uses the cashed maintenance and detections data to fill the track-data array.

        '''
        for ind, passes in enumerate(self._maintinence):
            if self._known[ind] and passes is not None:
                t_passes   = passes['t']
                snr_passes = passes['snr']

                for txi in range(len(t_passes)):
                    pas_inds = [ xi 
                        for xi, x in enumerate(t_passes[txi])
                        if x[0] >= t0 and x[0] <= t1
                    ]

                    n_tracks = len(pas_inds)

                    _tracks = np.empty((n_tracks,), dtype=self._track_format)
                    for track_id, pas_ind in enumerate(pas_inds):

                        SNR_l = np.array([ x[0] for x in snr_passes[txi][pas_ind] ])
                        SNR_tl = [ x[1] for x in snr_passes[txi][pas_ind] ]
                        baseline_n = np.sum( SNR_l > radar._tx[txi].enr_thresh )

                        _tracks[track_id]['t0'] = t_passes[txi][pas_ind][0]
                        _tracks[track_id]['dt'] = t_passes[txi][pas_ind][1] - t_passes[txi][pas_ind][0]
                        _tracks[track_id]['tracklet'] = False
                        _tracks[track_id]['tracklet_index'] = 0
                        _tracks[track_id]['tracklet_len'] = 0
                        _tracks[track_id]['SNRdB'] = 10.0*np.log10(np.max(SNR_l))
                        _tracks[track_id]['SNRdB-t'] = SNR_tl[np.argmax(SNR_l)]
                        _tracks[track_id]['index'] = ind
                        _tracks[track_id]['baselines'] = baseline_n
                        _tracks[track_id]['type'] = 'tracking'

                    self.add_tracks(num = n_tracks, data = _tracks)

        for ind, detections in enumerate(self._detections):
            if not self._known[ind] and detections is not None:

                for txi, tx_det in enumerate(detections):
                    det_inds = np.array([ xi
                        for xi, x in enumerate(tx_det["tm"])
                        if x >= t0 and x <= t1
                    ])

                    if radar_control is not None:
                        det_times = np.array(tx_det["tm"])
                        det_times = det_times[det_inds]
                        _control_check = radar_control(det_times)
                        det_times = det_times[_control_check]
                        det_inds = det_inds[_control_check]

                        #now these points are all detected and recored since we have the data.
                        #The first makes detection point, the rest makes the "initial" tracklet
                        #without any follow-up, that comes later in scheduling.

                        first_det = np.argmin(det_times)

                        self.add_tracklet(
                            track_id = len(self.tracks),
                            t = det_times,
                            index = ind,
                            fnames = [],
                            is_prior = False,
                        )
                        _track_data = {}
                        _track_data['tracklet'] = True
                        _track_data['tracklet_len'] = len(det_times)
                        _track_data['tracklet_index'] = len(self.tracklets) - 1

                        det_inds = [det_inds[first_det]]


                    n_tracks = len(det_inds)

                    _tracks = np.empty((n_tracks,), dtype=self._track_format)
                    for track_id, det_ind in enumerate(det_inds):

                        SNR_l = np.array(tx_det['snr'][det_ind])

                        baseline_n = np.sum( SNR_l > radar._tx[txi].enr_thresh )
                        
                        _tracks[track_id]['t0'] = tx_det["tm"][det_ind]
                        _tracks[track_id]['dt'] = tx_det["t1"][det_ind] - tx_det["tm"][det_ind]
                        if radar_control is None:
                            _tracks[track_id]['tracklet'] = False
                            _tracks[track_id]['tracklet_index'] = 0
                            _tracks[track_id]['tracklet_len'] = 0
                        else:
                            _tracks[track_id]['tracklet'] = _track_data['tracklet']
                            _tracks[track_id]['tracklet_index'] = _track_data['tracklet_index']
                            _tracks[track_id]['tracklet_len'] = _track_data['tracklet_len']
                        _tracks[track_id]['SNRdB'] = 10.0*np.log10(np.max(SNR_l))
                        _tracks[track_id]['SNRdB-t'] = tx_det["tm"][det_ind]
                        _tracks[track_id]['index'] = ind
                        _tracks[track_id]['baselines'] = baseline_n
                        _tracks[track_id]['type'] = 'scan'

                    self.add_tracks(num = n_tracks, data = _tracks)


    def __getitem__(self, key):
        if isinstance(key, int):
            ret_dict = {}
            ret_dict[self._data_format[0]] = self._oids[key]
            ret_dict[self._data_format[1]] = self._known[key]
            ret_dict[self._data_format[2]] = self._t0_unix[key]
            _tmp_tracks = self.tracks[self.tracks['index'] == key]
            ret_dict[self._data_format[3]] = len(_tmp_tracks)
            _tmp_tracklets = _tmp_tracks['tracklet_index'][_tmp_tracks['tracklet']]
            ret_dict[self._data_format[4]] = len(_tmp_tracklets)
            ret_dict[self._data_format[5]] = self._discovered[key]
            if self._discovered[key]:
                ret_dict[self._data_format[6]] = self._discovery_time[key]
            else:
                ret_dict[self._data_format[6]] = None
            return ret_dict
        else:
            raise Exception('Key type "{}" not recognized'.format(type(key)))


    def maintain(self, inds):
        '''Set object(s) to be maintained.
        '''
        if isinstance(inds, slice) or isinstance(inds, int):
            self._known[inds] = True
        else:
            for ind in inds:
                self._known[ind] = True


    def save(self, fname):
        '''Save all data related to the catalog to a hdf5 file.
        '''
        self._calculations()

        with h5py.File(fname,"w") as hf:

            hf.create_dataset('_discovered', data=self._discovered)
            hf.create_dataset('_discovery_track', data=self._discovery_track)
            hf.create_dataset('_discovery_time', data=self._discovery_time)
            hf.create_dataset('tracks', data=self.tracks)
            hf.create_dataset('_known', data=self._known)

            _dtype = copy.deepcopy(self._prior_dtype)
            _dtype[0] = ('date','float64')
            _priors = np.empty(self.priors.shape, dtype=_dtype)
            for tup in self._prior_dtype:
                if tup[0] == 'date':
                    _priors[tup[0]] = dpt.npdt2unix(self.priors[tup[0]])
                else:
                    _priors[tup[0]] = self.priors[tup[0]]

            hf.create_dataset('priors', data=_priors)

            hf.create_dataset('_tracklet_statistics', data=self._tracklet_statistics)
            hf.create_dataset('_tracks_summary', data=self._tracks_summary)
            hf.create_dataset('_tracking_summary', data=self._tracking_summary)
            hf.create_dataset('_scan_summary', data=self._scan_summary)


            grp_maint = hf.create_group("_maintinence")
            for ind, passes in enumerate(self._maintinence):
                if passes is not None:
                    tmp_grp = grp_maint.create_group('{}'.format(ind))

                    tmp_grp.create_dataset(
                        'snr',
                        data=np.array(passes['snr'], dtype=np.float),
                    )
                    tmp_grp.create_dataset(
                        't',
                        data=np.array(passes['t'], dtype=np.float),
                    )


            grp_det = hf.create_group("_detections")
            for ind, dets in enumerate(self._detections):
                if dets is not None:
                    tmp_grp = grp_det.create_group('{}'.format(ind))

                    for txi, det_dat in enumerate(dets):
                        tx_grp = tmp_grp.create_group('{}'.format(txi))

                        for key, val in det_dat.items():
                            tx_grp.create_dataset(
                                '{}'.format(key),
                                data=np.array(val, dtype=np.float),
                            )


            grp_trc = hf.create_group("tracklets")
            grp_trc.create_dataset(
                'track_id',
                data=np.array(
                    [x['track_id'] for x in self.tracklets],
                    dtype=np.int64,
                ),
            )
            grp_trc.create_dataset(
                'index',
                data=np.array(
                    [x['index'] for x in self.tracklets],
                    dtype=np.int64,
                ),
            )
            grp_trc.create_dataset(
                'is_prior',
                data=np.array(
                    [x['is_prior'] for x in self.tracklets],
                    dtype=np.bool,
                ),
            )
            for ind, tracklet in enumerate(self.tracklets):
                grp_trc.create_dataset(
                    '{}/t'.format(ind),
                    data=tracklet['t'],
                )
                grp_trc.create_dataset(
                    '{}/fnames'.format(ind),
                    data=np.array(tracklet['fnames']),
                )


    @classmethod
    def from_file(cls, population, fname):
        cat = cls(population)
        cat.load(fname)
        return cat


    def add_prior(self, index, state, cov, date):
        
        _priors = np.empty((1,), dtype=self.priors.dtype)
        _priors[0]['x'] = state[0]
        _priors[0]['y'] = state[1]
        _priors[0]['z'] = state[2]
        _priors[0]['vx'] = state[3]
        _priors[0]['vy'] = state[4]
        _priors[0]['vz'] = state[5]
        _priors[0]['date'] = date
        _priors[0]['index'] = index
        _priors[0]['cov'] = cov 
        self.priors = np.append(self.priors, _priors)
    
    def load(self, fname):
        '''Create a instance of Catalogue using a saved file and a population.
        '''
        with h5py.File(fname,"r") as hf:

            self._discovered = hf['_discovered'].value
            self._discovery_track = hf['_discovery_track'].value
            self._discovery_time = hf['_discovery_time'].value
            self.tracks = hf['tracks'].value
            self._known = hf['_known'].value

            _priors = hf['priors'].value
            self.priors = np.empty(_priors.shape, dtype=self._prior_dtype)
            for tup in self._prior_dtype:
                if tup[0] == 'date':
                    self.priors[tup[0]] = dpt.unix2npdt(_priors[tup[0]])
                else:
                    self.priors[tup[0]] = _priors[tup[0]]
                
            self._tracklet_statistics = hf['_tracklet_statistics'].value
            self._tracks_summary = hf['_tracks_summary'].value
            self._tracking_summary = hf['_tracking_summary'].value
            self._scan_summary = hf['_scan_summary'].value

            grp_maint = hf['_maintinence/']
            for ind in range(self.size):
                if '{}'.format(ind) in grp_maint:
                    self._maintinence[ind] = {
                        'snr': grp_maint['{}/snr'.format(ind)].value.tolist(),
                        't': grp_maint['{}/t'.format(ind)].value.tolist(),
                    }

            grp_det = hf['_detections/']
            for ind in range(self.size):
                if '{}'.format(ind) in grp_det:
                    tmp_grp = grp_det['{}/'.format(ind)]

                    self._detections[ind] = [None]*len(tmp_grp)
                    for txi in tmp_grp:
                        self._detections[ind][int(txi)] = {}
                        for field in self._det_fields:
                            self._detections[ind][int(txi)][field] = tmp_grp['{}/{}'.format(txi,field)].value.tolist()

            grp_trc = hf["tracklets/"]
            _track_id = grp_trc['track_id'].value
            _index = grp_trc['index'].value
            _is_prior = grp_trc['is_prior'].value
            
            _len_tracklets = len(_track_id)
            self.tracklets = [None]*_len_tracklets

            for ind in range(_len_tracklets):
                self.tracklets[ind] = {}
                self.tracklets[ind]['track_id'] = _track_id[ind]
                self.tracklets[ind]['index'] = _index[ind]
                self.tracklets[ind]['t'] = grp_trc['{}/t'.format(ind)].value
                self.tracklets[ind]['fnames'] = grp_trc['{}/fnames'.format(ind)].value.tolist()
                self.tracklets[ind]['is_prior'] = _is_prior[ind]


    def __str__(self):
        _str = ''

        _str += '------ TRACKS ------'
        for ind, track in enumerate(cat.tracks):
            _str += '\n' + 'Track: {}'.format(ind)
            for field in cat.tracks.dtype.names:
                _str += '\n' +' - {}: {}'.format(field, track[field])

        _str += '\n'*2 + '------ TRACKLETS ------'
        for ind, tracklet in enumerate(cat.tracklets):
            _str += '\n' + 'Tracklet: {}'.format(ind)
            for field in tracklet:
                _str += '\n' + ' - {}: {}'.format(field, tracklet[field])

        _str += '\n'*2 + '------ OBJECTS ------'
        for ind in range(cat.size):
            _str += '\n' + '== Object {} =='.format(ind)
            for key, val in cat[ind].items():
                _str += '\n' + '{:<20}: {}'.format(key, val)

        return _str


    def maintinece_summary(self):
        '''Compute summary statistics about maintenance.
        '''

        maintinence_times = []
        maintinence_SNRdb = []
        maintinence_track_time = []
        maintinence_index = []
        maint_obj_n = 0
        maint_objs = 0
        for index, passes in enumerate(self._maintinence):
            if passes is not None:
                maint_objs +=1
                t   = passes['t']
                snr = passes['snr']
                for txi in range(len(t)):
                    tx_t  =   t[txi]
                    tx_snr= snr[txi]
                    n_maint=len(tx_t)
                    maint_obj_n += n_maint
                    for m_id in range(n_maint):
                        SNR_l = [ x[0] for x in tx_snr[m_id] ]
                        SNR_tl = [ x[1] for x in tx_snr[m_id] ]

                        maintinence_index.append(index)
                        maintinence_times.append( SNR_tl[np.argmax(SNR_l)]/3600.0 )
                        maintinence_SNRdb.append(10.0*np.log10(np.max( SNR_l )))
                        maintinence_track_time.append(( tx_t[m_id][1] - tx_t[m_id][0])/60.0 )
        

        self._tracking_summary = np.empty((len(maintinence_times),), dtype=self._tracking_summary_dtype)
        self._tracking_summary['index'] = maintinence_index
        self._tracking_summary['peak-SNR-time'] = maintinence_times
        self._tracking_summary['peak-SNRdB'] = maintinence_SNRdb
        self._tracking_summary['tracking-time'] = maintinence_track_time


        unknown_n = np.sum( 1 for x in self._known if not x )
        known_n = np.sum( 1 for x in self._known if x )

        if maint_objs > 0:
            print('Total %i of %i known objects maintained'%(maint_objs, known_n ))
            print('Total %.2f %% of objects maintained an average of %.2f tracklets per object.' % \
             (float(maint_objs)/known_n*100.0, maint_obj_n/maint_objs, ) )
        else:
            print('No maintained objects')


    def maintinece_summary_plot(self, save_folder = None):
        if len(self._tracking_summary) > 0:
            opts = {}
            opts['title'] = "Temporal distribution of peak SNR of possible tracklets"
            opts['xlabel'] = "Detection time past $t_0$ [h]"
            opts['bin_size'] = 1.0
            if save_folder is not None:
                opts['save'] = save_folder +'/possible_maintinence_times.jpg'
            else:
                opts['show'] = False

            fig, ax = dpt.hist(self._tracking_summary['peak-SNR-time'], **opts)
            
            opts = {}
            opts['title'] = "Peak SNR distribution of possible maintinence"
            opts['xlabel'] = "Peak SNR [dB]"
            if save_folder is not None:
                opts['save'] = save_folder +'/possible_maintinence_snr_dist.jpg'
            else:
                opts['show'] = False
            
            fig, ax = dpt.hist(self._tracking_summary['peak-SNRdB'], **opts)
            
            opts = {}
            opts['title'] = "Distribution of possible tracking time for maintinence"
            opts['xlabel'] = "Tracking time [min]"
            if save_folder is not None:
                opts['save'] = save_folder +'/possible_maintinence_track_lengths_dist.jpg'
            else:
                opts['show'] = False

            fig, ax = dpt.hist(self._tracking_summary['tracking-time'], **opts)

    
    def detection_summary(self):
        detection_index = []
        detection_times = []
        detection_SNRdb = []
        detection_track_time = []
        det_obj_n = 0
        det_objs = 0
        for index, detections in enumerate(self._detections):
            if detections is not None:
                det_objs +=1
                for txi, tx in enumerate(detections):
                        n_dets=len(tx["tm"])
                        det_obj_n += n_dets
                        for det_idx in range(n_dets):
                            detection_index.append(index)
                            detection_times.append(tx["tm"][det_idx]/3600.0)
                            detection_SNRdb.append(10.0*np.log10(np.max(tx["snr"][det_idx])))
                            detection_track_time.append((tx["t1"][det_idx] - tx["tm"][det_idx])/60.0 )

        unknown_n = np.sum( 1 for x in self._known if not x )
        known_n = np.sum( 1 for x in self._known if x )

        self._scan_summary = np.empty((len(detection_index),), dtype=self._scan_summary_dtype)
        self._scan_summary['index'] = detection_index
        self._scan_summary['detection-time'] = detection_times
        self._scan_summary['SNRdB'] = detection_SNRdb
        self._scan_summary['tracking-time'] = detection_track_time

        if det_objs > 0:
            print('Total %i of %i possible objects detected'%(det_objs,unknown_n))
            print('Total %.2f %% of objects detected an average of %.2f times per detected object.' \
             % (float(det_objs)/unknown_n*100.0, det_obj_n/det_objs, ) )
        else:
            print('No detected objects')

    def detection_summary_plot(self, save_folder = None):
        if len(self._scan_summary) > 0:

            opts = {}
            opts['title'] = "Temporal distribution of possible detections"
            opts['xlabel'] = "Detection time past $t_0$ [h]"
            opts['bin_size'] = 1.0
            if save_folder is not None:
                opts['save'] = save_folder +'/possible_detection_times.jpg'
            else:
                opts['show'] = False

            fig, ax = dpt.hist(self._scan_summary['detection-time'], **opts)
            
            opts = {}
            opts['title'] = "Peak SNR distribution of possible detections"
            opts['xlabel'] = "Peak SNR [dB]"
            if save_folder is not None:
                opts['save'] = save_folder +'/possible_snr_dist.jpg'
            else:
                opts['show'] = False

            fig, ax = dpt.hist(self._scan_summary['SNRdB'], **opts)
            
            opts = {}
            opts['title'] = "Distribution of possible tracking time after detection"
            opts['xlabel'] = "Tracking time [min]"
            if save_folder is not None:
                opts['save'] = save_folder +'/possible_track_t_dist.jpg'
            else:
                opts['show'] = False

            fig, ax = dpt.hist(self._scan_summary['tracking-time'], **opts)


    def maintenance_tracks_plot(self, save_folder = None):

        tracks_tracking = self.tracks[self.tracks['type'] == 'tracking']

        if len(tracks_tracking) > 0:

            opts = {}
            opts['title'] = "Temporal distribution of maintenance tracklets"
            opts['xlabel'] = "Tracklet peak SNR time past $t_0$ [h]"
            opts['bin_size'] = 1.0
            if save_folder is not None:
                opts['save'] = save_folder +'/maintenance_times.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(tracks_tracking['SNRdB-t']/3600.0, **opts)
            
            opts = {}
            opts['title'] = "Peak SNR distribution of maintenance tracklets"
            opts['xlabel'] = "Peak SNR [dB]"
            if save_folder is not None:
                opts['save'] = save_folder +'/maintenance_snr_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(tracks_tracking['SNRdB'], **opts)

            opts = {}
            opts['title'] = "Distribution of scheduled maintenance time"
            opts['xlabel'] = "Tracking window [min]"
            if save_folder is not None:
                opts['save'] = save_folder +'/maintenance_track_dt_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(tracks_tracking['dt']/60.0, **opts)

            opts = {}
            opts['title'] = "Distribution of scheduled measurements per maintenance tracklets"
            opts['xlabel'] = "Tracklet points per tracklet [1]"
            if save_folder is not None:
                opts['save'] = save_folder +'/maintenance_tracklet_point_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(tracks_tracking['tracklet_len'], **opts)

            opts = {}
            opts['title'] = "Distribution of maintenance track baselines"
            opts['xlabel'] = "Number of baselines [1]"
            opts['bins'] = [1,2,3,4]
            if save_folder is not None:
                opts['save'] = save_folder +'/maintenance_baseline_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(tracks_tracking['baselines'], **opts)


    def track_statistics(self):

        self._tracks_summary = np.empty((self.size, ), dtype=self._tracks_summary_dtype)
        self._tracklet_statistics = np.empty((len(self.tracklets), ), dtype=self._tracklet_statistics_dtype)
        
        for ind in range(self.size):
            if self._known[ind]:
                tr_type = 'tracking'
            else:
                tr_type = 'scan'

            self._tracks_summary[ind]['index'] = ind
            self._tracks_summary[ind]['tracks'] = np.sum(self.tracks['index'] == ind)
            self._tracks_summary[ind]['type'] = tr_type
            self._tracks_summary[ind]['tracklets']  = np.sum(np.logical_and(self.tracks['index'] == ind, self.tracks['tracklet']))
            _tracklet_lens = [len(x['t']) for x in self.tracklets if self.tracks[x['track_id']]['index'] == ind]
            self._tracks_summary[ind]['tracklet-len-sum'] = np.sum(_tracklet_lens)
            self._tracks_summary[ind]['tracklet-len-mean'] = np.mean(_tracklet_lens)
            self._tracks_summary[ind]['tracklet-len-std'] = np.std(_tracklet_lens)
            _tracklet_SNRdBs = self.tracks[self.tracks['index'] == ind]['SNRdB']
            self._tracks_summary[ind]['peak-SNRdB-mean'] = np.mean(_tracklet_SNRdBs)
            self._tracks_summary[ind]['peak-SNRdB-std'] = np.std(_tracklet_SNRdBs)



        for ind, tracklet in enumerate(self.tracklets):
            trid = tracklet['track_id']
            track_dt = self.tracks[trid]['dt']
            rel_t = tracklet['t'] - self.tracks[trid]['SNRdB-t']

            self._tracklet_statistics[ind]['index'] = tracklet['index']
            self._tracklet_statistics[ind]['track-id'] = tracklet['track_id']

            self._tracklet_statistics[ind]['points-deviation-mean'] = np.mean(rel_t)
            self._tracklet_statistics[ind]['points-deviation-std'] = np.std(rel_t)
            self._tracklet_statistics[ind]['points-deviation-skew'] = scipy.stats.skew(rel_t)
            self._tracklet_statistics[ind]['points'] = len(rel_t)
            self._tracklet_statistics[ind]['peak-SNRdB'] = self.tracks[trid]['SNRdB']
            self._tracklet_statistics[ind]['span'] = np.max(tracklet['t']) - np.min(tracklet['t'])
            if track_dt > 0:
                self._tracklet_statistics[ind]['points-deviation-normalized-mean'] = np.mean(rel_t/track_dt)
                self._tracklet_statistics[ind]['points-deviation-normalized-std'] = np.std(rel_t/track_dt)
                self._tracklet_statistics[ind]['normalized-span'] = (np.max(tracklet['t']) - np.min(tracklet['t']))/track_dt
            else:
                self._tracklet_statistics[ind]['points-deviation-normalized-mean'] = 0.0
                self._tracklet_statistics[ind]['points-deviation-normalized-std'] = 0.0
                self._tracklet_statistics[ind]['normalized-span'] = 0.0



    def track_statistics_plot(self, save_folder = None):

        if len(self._tracklet_statistics) > 0:
            opts = {}
            opts['title'] = "Tracklet points vs peak SNR"
            opts['xlabel'] = "Tracklet points"
            opts['ylabel'] = "Track peak SNR [dB]"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_vs_snr.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.scatter(
                self._tracklet_statistics['points'],
                self._tracklet_statistics['peak-SNRdB'],
                **opts
            )

            opts = {}
            opts['title'] = "Tracklet points vs available passes"
            opts['xlabel'] = "Mean number of tracklet points per pass"
            opts['ylabel'] = "Available passes"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_vs_passes.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.scatter(
                self._tracks_summary['tracklet-len-mean'],
                self._tracks_summary['tracks'],
                **opts
            )

            opts = {}
            opts['title'] = "Total number of tracklet points per object"
            opts['xlabel'] = "Total data points over all tracklets"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_total.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(self._tracks_summary['tracklet-len-sum'], **opts)

            opts = {}
            opts['title'] = "Normalized mean tracklet point distribution around track peak SNR"
            opts['xlabel'] = "Normalized mean displacement $E[t_i - t_m]/\Delta t$ [1]"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_norm_mean.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(self._tracklet_statistics['points-deviation-normalized-mean'], **opts)
            
            opts = {}
            opts['title'] = "Normalized standard deviation tracklet point distribution around track peak SNR"
            opts['xlabel'] = "Normalized standard deviation of displacement $\sigma[t_i]/\Delta t$ [1]"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_norm_std.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(self._tracklet_statistics['points-deviation-normalized-std'], **opts)

            opts = {}
            opts['title'] = "Normalized tracklet span distribution"
            opts['xlabel'] = "Tracklet span $(max[t_i] - min[t_i])/\Delta t$ [1]"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_norm_span.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(self._tracklet_statistics['normalized-span'], **opts)

            opts = {}
            opts['title'] = "Mean tracklet point distribution around track peak SNR"
            opts['xlabel'] = "Mean displacement $E[t_i - t_m]$ [min]"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_mean.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(self._tracklet_statistics['points-deviation-mean']/60.0, **opts)
            
            opts = {}
            opts['title'] = "Standard deviation tracklet point distribution around track peak SNR"
            opts['xlabel'] = "Standard deviation of displacement $\sigma[t_i]$ [min]"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_std.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(self._tracklet_statistics['points-deviation-std']/60.0, **opts)

            opts = {}
            opts['title'] = "Skewness tracklet point distribution around track peak SNR"
            opts['xlabel'] = "Skewness of displacement $\gamma[t_i - t_m]$ [min]"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_sqew.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(self._tracklet_statistics['points-deviation-skew']/60.0, **opts)

            opts = {}
            opts['title'] = "Tracklet span distribution"
            opts['xlabel'] = "Tracklet span $max[t_i] - min[t_i]$ [min]"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_span.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(self._tracklet_statistics['span'], **opts)
            
            opts = {}
            opts['title'] = "Tracklet point number distribution"
            opts['xlabel'] = "Number of tracklet points $N[t_i]$ [1]"
            if save_folder is not None:
                opts['save'] = save_folder +'/tracklet_points_num.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(self._tracklet_statistics['points'], **opts)



    def detection_tracks_plot(self, save_folder = None):

        _tr_scan = np.logical_and(self.tracks['type'] == 'scan', self.tracks['tracklet'])
        tracks_scan = self.tracks[_tr_scan]

        if len(tracks_scan) > 0:

            detection_times = []
            detection_times_all = []
            for ind in range(self.size):
                _test = np.logical_and(_tr_scan, self.tracks['index'] == ind)
                if np.any(_test):
                    _det_times = self.tracks[_test]['SNRdB-t']
                    detection_times.append(np.min(_det_times))
                    for _det in self.tracks[_test]:
                        detection_times_all += self.tracklets[_det['tracklet_index']]['t'].tolist()

            detection_times = np.array(detection_times, dtype=self._type)
            detection_times_all = np.array(detection_times_all, dtype=self._type)

            opts = {}
            opts['title'] = "Cumulative catalogue buildup [unique objects]"
            opts['xlabel'] = "Time past $t_0$ [h]"
            opts['cumulative'] = True
            if save_folder is not None:
                opts['save'] = save_folder +'/catalogue_buildup.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(detection_times/3600.0, **opts)


            opts = {}
            opts['title'] = "Temporal distribution of detections [unique objects]"
            opts['xlabel'] = "Detection time past $t_0$ [h]"
            opts['bin_size'] = 1.0
            if save_folder is not None:
                opts['save'] = save_folder +'/discovery_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(detection_times/3600.0, **opts)
            
            opts = {}
            opts['title'] = "Temporal distribution of measurements of unknown objects"
            opts['xlabel'] = "Time past $t_0$ [h]"
            opts['bin_size'] = 1.0
            if save_folder is not None:
                opts['save'] = save_folder +'/discovery_measurements_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(detection_times_all/3600.0, **opts)
            
            opts = {}
            opts['title'] = "Peak SNR distribution of detections"
            opts['xlabel'] = "Peak SNR [dB]"
            if save_folder is not None:
                opts['save'] = save_folder +'/discovery_snr_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(tracks_scan['SNRdB'], **opts)

            opts = {}
            opts['title'] = "Distribution of possible follow-up time"
            opts['xlabel'] = "Tracking window [min]"
            if save_folder is not None:
                opts['save'] = save_folder +'/discovery_track_dt_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(tracks_scan['dt']/60.0, **opts)

            opts = {}
            opts['title'] = "Distribution of scheduled follow-up points per tracklet"
            opts['xlabel'] = "Tracklet points per tracklet [1]"
            if save_folder is not None:
                opts['save'] = save_folder +'/discovery_tracklet_point_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(tracks_scan['tracklet_len'], **opts)

            opts = {}
            opts['title'] = "Distribution of discovery baselines"
            opts['xlabel'] = "Number of baselines [1]"
            opts['bins'] = [1,2,3,4]
            if save_folder is not None:
                opts['save'] = save_folder +'/discovery_baseline_dist.jpg'
            else:
                opts['show'] = False
            fig, ax = dpt.hist(tracks_scan['baselines'], **opts)

if __name__=='__main__':

    import population_library as plib

    pop = plib.master_catalog()
    pop.delete(slice(4,None))

    cat = Catalogue(pop)

    data = [
        [0, 12, False, 0, 0, 123, 8, 2, 3, 'scan'],
        [2.2, 12, True, 0, 10, 13, 8.4, 3, 3, 'tracking'],
    ]
    cat.tracklets.append({
            'track_id': 1,
            't': np.linspace(3,4,num=10),
            'index': 3,
            'fnames': ['this_tdm_0_{}.tdm'.format(rxi) for rxi in range(3)],
        }
    )
    cat.add_tracks(2, data=data)
    cat._detections[2] = [{"t0":[0],"t1":[12],"snr":[32],'tm':[9], "range":[300e3], "range_rate":[3e3], "tx_gain":[12e3], "rx_gain":[11e3], "on_axis_angle":[0.0]}]

    str_cat = str(cat)
    print(str_cat)

    fname = '/home/danielk/IRF/E3D_PA/tmp/test/test_cat.h5'
    cat.save(fname)

    cat2 = Catalogue.from_file(pop, fname)
    str_cat2 = str(cat2)
    print(str_cat2)

    assert str_cat == str_cat2
