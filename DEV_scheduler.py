import scheduler_library

data = {}
data['peak_snr'] = 34
data['tracklets'] = [3,5,10]
data['dt'] = 20
data['N'] = 44
data['source'] = 'track'

config = {}

config['N_on'] = True
config['dt_on'] = True
config['tracklets_on'] = True
config['peak_snr_on'] = True
config['source_on'] = True

config['N_rate'] = 50.
config['dt_sigma'] = 5*1.*60.
config['dt_offset'] = -1.*60. #shifts to later than max
config['dt_sqew'] = -0.8 # > 0 = initial trail longer
config['tracklets_scale'] = 15.
config['peak_snr_rate'] = 50.
config['track-scan_ratio'] = 0.5

config['N_scale'] = 1.
config['dt_scale'] = 5.
config['tracklets_rate'] = 2.
config['peak_snr_scale'] = 0.5
config['tracklet_completion_rate'] = 20.0


X = scheduler_library.que_value_dyn_v2(data, config, debug=True)