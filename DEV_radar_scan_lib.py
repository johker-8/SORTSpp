import sys

import numpy as np
import matplotlib.pyplot as plt

from radar_scans import plot_radar_scan, plot_radar_scan_movie
import radar_scan_library as rslib

def make_menu(scan_list):
    while True:
        print('\n'*100)
        print('-'*10 + 'SCAN PLOT MENU' + '-'*10)
        for ind, sc_item in enumerate(scan_list):
            print( '{:<4}: {}'.format(ind, sc_item[0]) )
        print( '{:<4}: Exit'.format('exit') )
        print( '{:<4}: Exit'.format('all') )
        ans = raw_input('$ ')
        if ans.strip().lower() == 'exit':
            break
        elif ans.strip().lower() == 'all':
            for fun in [scan[1] for scan in scan_list]:
                fun()
        else:
            fun = scan_list[int(ans)][1]
            fun()
        



def beampark_plot():
    scan = rslib.beampark_model(az = 0., el = 45.0, lat = 69., lon = 19., alt = 150.)
    plot_radar_scan(scan, earth=True)
    plt.show()

def n_beampark_plot():
    az_points = np.arange(0.,360.,60.).tolist();
    el_points = np.linspace(80., 90., num=len(az_points)).tolist();
    scan = rslib.n_const_pointing_model(az = az_points, el = el_points, lat = 69., lon = 19., alt = 150., dwell_time=0.1)
    
    plot_radar_scan(scan, earth=True)
    plt.show()
    
def n_beampark_plot_mov():
    az_points = np.arange(0.,360.,60.).tolist();
    el_points = np.linspace(80., 90., num=len(az_points)).tolist();
    scan = rslib.n_const_pointing_model(az = az_points, el = el_points, lat = 69., lon = 19., alt = 150., dwell_time=0.1)
    
    plot_radar_scan_movie(scan, earth=True, rotate=False)
    plt.show()

def sph_random_plot():
    scan = rslib.sph_rng_model(lat = 69., lon = 19., alt = 150., min_el = 30, dwell_time = 0.1)
    plot_radar_scan(scan, earth=True)
    plt.show()

def sph_random_plot_mov():
    scan = rslib.sph_rng_model(lat = 69., lon = 19., alt = 150., min_el = 30, dwell_time = 0.1)
    plot_radar_scan_movie(scan, earth=True, rotate=True)
    plt.show()

def ew_fence_plot():
    scan = rslib.ew_fence_model(lat = 69., lon = 19., alt = 150.)
    plot_radar_scan(scan, earth=True)
    plt.show()

def ns_fence_plot():
    scan = rslib.ns_fence_model(lat = 69., lon = 19., alt = 150.)
    plot_radar_scan(scan, earth=True)
    plt.show()

def ew_fence_plot_mov():
    scan = rslib.ew_fence_model(lat = 69., lon = 19., alt = 150.)
    plot_radar_scan_movie(scan, earth=True)
    plt.show()

def ns_fence_plot_mov():
    scan = rslib.ns_fence_model(lat = 69., lon = 19., alt = 150.)
    plot_radar_scan_movie(scan, earth=True)
    plt.show()

def ns_fence_rng_plot_mov():
    scan = rslib.ns_fence_rng_model(lat = 69., lon = 19., alt = 150.)
    plot_radar_scan_movie(scan, earth=True)
    plt.show()



def dyn_n_beampark_plot_mov():
    az_points = [0.0]*20 + [180.]*20
    el_points = np.linspace(0,90, num=20).tolist()
    el_points += el_points[::-1]
    
    dwells = np.linspace(0.1, 0.8, num=40)
    scan = rslib.n_dyn_dwell_pointing_model(az = az_points, el = el_points, dwells = dwells, lat = 69., lon = 19., alt = 150.)
    plot_radar_scan_movie(scan, earth=True)
    plt.show()

def interleaved_exp_plot_mov():
    el_points_fence = rslib.calculate_fence_angles(min_el=30.0,angle_step=1.0)
    az_points_fence = [90]*len(el_points_fence[el_points_fence > 90]) + [180]*len(el_points_fence[el_points_fence <= 90])
    el_points_fence[el_points_fence > 90] = 180-el_points_fence[el_points_fence > 90]
    
    dwells = []
    az_points = []
    el_points = []
    for ind in range(len(el_points_fence)):
        dwells.append(0.4)
        az_points.append(0.0)
        el_points.append(-90.0)
        dwells.append(0.1)
        az_points.append(az_points_fence[ind])
        el_points.append(el_points_fence[ind])
    
    scan = rslib.n_dyn_dwell_pointing_model(az=az_points, el=el_points, dwells=dwells, lat = 69., lon = 19., alt = 150.)
    plot_radar_scan_movie(scan, earth=True)
    plt.show()

def flat_grid_plot():
    scan = rslib.flat_grid_model(lat = 69., lon = 19., alt = 150., n_side = 5, height = 300e3, side_len = 200e3, dwell_time = 0.1)
    plot_radar_scan(scan, earth=True)
    plt.show()


def leak_proof():

    beam_width = 1.26 #deg
    max_sat_speed = 8e3 #m/s
    leak_proof_at = 350e3 #altitude

    els = ['90.0', '88.73987398739877', '87.4357183693167', '86.1720061939742', '84.90239189256025', '83.628529014365', '82.35204237515919', '81.07452478944677', '79.77710211752762', '78.47281161299796', '77.16391480976668', '75.83375939658995', '74.49527977424731', '73.13358804398229', '71.753175185289', '70.3584796645483', '68.93771910430019', '67.48130276816576', '65.99689473774336', '64.48487395668005', '62.93289943117691', '61.33878768753357', '59.69647098103616', '57.99766296011879', '56.23923388331565', '54.40755219009088', '52.44010673902446', '50.4247836215483', '48.29222295539205', '46.01278403125778', '43.55617730018973', '40.8717857555977', '37.886094818016815', '34.478961143266076', '30.0']
    
    els = els[:5]
    print('MIN_EL = {:.4f}'.format(float(els[-1])))
    els = [float(el) for el in els]
    azs = [-180.0]*(len(els)-1) + [0.0]*len(els)
    els = els[1:][::-1] + els

    dwell_time = []
    for el in els:

        beam_arc_len = np.radians(beam_width)*leak_proof_at/np.sin(np.radians(el))
        sat_traverse_time = beam_arc_len/max_sat_speed #also max scan time
        #print('sat_traverse_time: {:.4f} s'.format(sat_traverse_time))
        dwell_time.append( sat_traverse_time/float(len(azs)) )

    print(dwell_time)

    scan = rslib.n_dyn_dwell_pointing_model(
        az = azs,
        el = els,
        dwells = dwell_time,
        lat = 69., lon = 19., alt = 150.,
    )
    plot_radar_scan(scan, earth=True)
    plt.show()

if __name__=='__main__':
    scan_list = [
        ('Beampark', beampark_plot),
        ('n-point beampark', n_beampark_plot),
        ('n-point beampark [Movie]', n_beampark_plot_mov),
        ('Uniform random spherical scan', sph_random_plot),
        ('Uniform random spherical scan [Movie]', sph_random_plot_mov),
        ('East-west fence scan', ew_fence_plot),
        ('North-south fence scan', ns_fence_plot),
        ('Random North-south fence scan movie', ns_fence_rng_plot_mov),
        ('East-west fence scan [Movie]', ew_fence_plot_mov),
        ('North-south fence scan [Movie]', ns_fence_plot_mov),
        ('Dynamic n-point beampark [Movie]', dyn_n_beampark_plot_mov),
        ('Interleaved no-data-access experiments [Movie]', interleaved_exp_plot_mov),
        ('Flat grid scan', flat_grid_plot),
        ('Leak proof fence scan', leak_proof),
    ]
    make_menu(scan_list)