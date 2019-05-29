#!/usr/bin/env python

'''Collection of classes and functions related to constructing a radar system scheduler.

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import copy

import plothelp
import rewardf_library

def scheduler_TEMPLATE(catalogue, radar, parameters, t0, t1, **kwargs):
    pass



def _binary_radar_control(t, on_time, off_time):
    tm = t % (on_time+off_time)
    return tm <= on_time



def dynamic_scheduler(catalogue, radar, parameters, t0, t1, **kwargs):
    '''Dynamic scheduler

    '''

    reward_function = kwargs.setdefault('reward_function', rewardf_library.rewardf_exp_peak_SNR_tracklet_len)
    reward_function_config = kwargs.setdefault('reward_function_config', {
            'sigma_t': 2.0*60.0,
            'lambda_N': 10.0
        })
    logger = kwargs.setdefault('logger', None)


    on_time = parameters.SST_slices * parameters.coher_int_t
    off_time = parameters.Interleaving_slices * parameters.interleaving_time_slice

    radar_control = lambda t: _binary_radar_control(t, on_time, off_time)

    catalogue.compile_tracks(radar, t0, t1, radar_control = radar_control)

    if logger is not None:
        logger.info('Raw track count: {}'.format(len(catalogue.tracks)))

    t_sort = np.argsort(catalogue.tracks['t0']) #chronological order
    catalogue.tracks = catalogue.tracks[t_sort]

    if len(catalogue.tracks) == 0:
        logger.error('No schedule constructed, no tracks available.')
        return catalogue

    #these tracks will be deleted after scheduler analysis is complete
    keep_flag = np.full(catalogue.tracks.shape, True, dtype=np.bool)

    #lets go with a queue system according to track ID
    #we know that they are in chronological order
    queue = []
    queue_OID = []
    queue_dist = {}
    #we create a list of events, i.e. start to stop measurements
    # 0 = start
    # 1 = stop
    tr_t1 = catalogue.tracks['t0'] + catalogue.tracks['dt']
    t_events = [ (x, 0, ID) for ID,x in enumerate(catalogue.tracks['t0']) ]\
             + [ (x, 1, ID) for ID,x in enumerate(tr_t1) ]

    t_events.sort(key=lambda x: x[0])
    t_prev = t_events[0][0]

    t_first = t_events[0][0]
    t_last = t_events[-1][0]
    _done = 0
    _time_start = time.time()
    _frac = 1000
    _iter_len = len(t_events)
    _iter_now = 0

    obj_inds = catalogue.tracks['index']

    if parameters.tracking_f < 1e-3:
        t_events = []

    for t_now, event_type, ID in t_events:
        _iter_now += 1
        #now this queue will contain all track ID's that request data points 
        #at every point in time discretized according to events

        if len(queue) > 0 and t_prev != t_now:

            #create a timeline between the last event and this according to our time slices
            t_start = np.round(t_prev / (on_time+off_time))*(on_time+off_time)
            t_stop =  np.round(t_now  / (on_time+off_time))*(on_time+off_time)
            
            t = np.arange(t_start, t_stop, parameters.coher_int_t )

            #find witch times are possible due to radar control
            t = t[_binary_radar_control(t, on_time, off_time)]

            #distribute points to queue
            queue_reward = np.zeros((len(queue),))

            for ti in range(len(t)):
                #calculate the object queue position
                for qI, queue_entry in enumerate(queue):

                    queue_reward[qI] = reward_function(
                        t[ti],
                        catalogue.tracks[queue_entry],
                        reward_function_config,
                    )
                
                que_id = np.argmax(queue_reward)
                queue_dist[queue[que_id]].append(t[ti])

            '''
            if logger is not None:
                logger.debug('STEP {:.2f} h : handing out {} points'.format(t_now/3600.0, len(t)))
                for qI in range(len(queue)):
                    logger.debug('-- ID {} - que place {:.5f}: has {} points'.format(queue[qI],queue_reward[qI],len(queue_dist[queue[qI]])))
            '''
        #add and remove objects to queue
        if event_type == 1:
            #logger.info('Del: OID {} qI {} ID {}'.format(queue_OID[qI], qI, ID))
            qI = queue.index(ID)
            if len(queue_dist[ID]) > 0:
                #we have a tracklet, add to list
                if catalogue.tracks[ID]['tracklet']:
                    tmp_tracklet = catalogue.tracklets[catalogue.tracks[ID]['tracklet_index']]
                    tmp_tracklet['t'] = np.append(tmp_tracklet['t'], np.array(queue_dist[ID]), axis=0)
                    catalogue.tracks[ID]['tracklet_len'] += len(queue_dist[ID])
                else:
                    catalogue.add_tracklet(
                        track_id = ID,
                        t = np.array(queue_dist[ID]),
                        index = queue_OID[qI],
                        fnames = [],
                        is_prior = False,
                    )
                    catalogue.tracks[ID]['tracklet'] = True
                    catalogue.tracks[ID]['tracklet_len'] = len(queue_dist[ID])
                    catalogue.tracks[ID]['tracklet_index'] = len(catalogue.tracklets) - 1
                
            del queue_OID[qI]
            del queue[qI]
            del queue_dist[ID]


        if event_type == 0:
            #logger.info('Add: OID {} ID {}'.format(obj_inds[ID], ID))
            oid = obj_inds[ID]
            if ID not in queue:
                if oid not in queue_OID:
                    queue.append(ID)
                    queue_OID.append(oid)
                    queue_dist[ID] = []
                else:
                    keep_flag[ID] = False

        #logger.info('queue: {}'.format(queue))
        dt_now = (t_now - t_first)/(t_last - t_first)

        if int(dt_now*float(_frac)) >= _done:
            _done = int(dt_now*float(_frac)) + 1

            if logger is not None:
                _time_elaps = time.time() - _time_start

                logger.info('{:.2f} % of scheduling complete at {:.3f} h, estimated time left {:.3f} h'.format(
                    dt_now*100.0,
                    _time_elaps/3600.0,
                    _time_elaps/float(_iter_now)*float(_iter_len - _iter_now)/3600.0,
                ))


        t_prev = t_now

    if len(t_events) > 0:
        _old_index = np.arange(len(catalogue.tracks))
        _old_index = _old_index[keep_flag].tolist()
        _new_index = list(range(len(_old_index)))
        #need to recalibrate all "track_index"
        for tracklet in catalogue.tracklets:
            ind = _old_index.index(tracklet['track_id'])
            tracklet['track_id'] = _new_index[ind]

        catalogue.tracks = catalogue.tracks[keep_flag]

    return catalogue







##
## THESE SCHEDULERS ARE NOW OUTDATED AND CANOT FUNCTION, NEED TO CONFORM TO NEW STANDARD
##
#this only tracks for config['memory_track_time'] seconds
#then if this tracking is achived it starts scanning again and remembers the object!
#it is only focused on discovery so it wont do follow ups
def memory_static_discovery_sceduler(sim,config):
    cat = sim._catalogue
    cat_detections = cat._detections
    tracks = []
    discovered = [[False,0]]*len(cat_detections)
    tracks_rem = 0
    for oid,detections in enumerate(cat_detections):
        if detections is not None:
            for txi,tx in enumerate(detections):
                    n_dets=len(tx["t0"])
                    for det_idx in range(n_dets):
                        tracks.append( [tx["t0"][det_idx], tx["t1"][det_idx] - tx["t0"][det_idx], True, 10.0*np.log10(np.max(tx["snr"][det_idx])), oid ] )
    tracks.sort(key=lambda x: x[0]) #chronological order
    memory = []
    for tri in range(len(tracks)-1):
        if tracks[tri][2]:
            di = 1
            if tracks[tri][4] not in memory:
                if tracks[tri][1] < config['memory_track_time']:
                    tracks_overlap_check = tracks[tri][0]+tracks[tri][1] > tracks[tri+di][0]
                else:
                    memory.append(tracks[tri][4])
                    tracks_overlap_check = tracks[tri][0]+config['memory_track_time'] > tracks[tri+di][0]
                
                while tracks_overlap_check: #checks all possibilities
                    tracks[tri+di][2] = False
                    tracks_rem+=1
                    di+=1
                    if tri+di < len(tracks):
                        if tracks[tri][1] < config['memory_track_time']:
                            tracks_overlap_check = tracks[tri][0]+tracks[tri][1] > tracks[tri+di][0]
                        else:
                            tracks_overlap_check = tracks[tri][0]+config['memory_track_time'] > tracks[tri+di][0]
                    else:
                        tracks_overlap_check = False
            else:
                tracks[tri][2] = False
                tracks_rem+=1

    #go trough again and check for discoveries
    for tri in range(len(tracks)):
        #if detected and not yet discovered, set to discovered
        if tracks[tri][2] and (not discovered[tracks[tri][4]][0]):
            discovered[tracks[tri][4]] = [True,tri]

    scheduler_return = {}
    scheduler_return['tracks'] = tracks
    scheduler_return['discovered'] = discovered
    scheduler_return['removed_tracks'] = tracks_rem
    return scheduler_return


##
## THESE SCHEDULERS ARE NOW OUTDATED AND CANOT FUNCTION, NEED TO CONFORM TO NEW STANDARD
##
def isolated_static_discovery_sceduler(sim,config):
    cat = sim._catalogue
    cat_detections = cat._detections
    tracks = []
    discovered = [[False,0]]*len(cat_detections)
    tracks_rem = 0
    for oid,detections in enumerate(cat_detections):
        if detections is not None:
            for txi,tx in enumerate(detections):
                    n_dets=len(tx["t0"])
                    for det_idx in range(n_dets):
                        tracks.append( [tx["t0"][det_idx], tx["t1"][det_idx] - tx["t0"][det_idx], True, 10.0*np.log10(np.max(tx["snr"][det_idx])), oid ] )
    tracks.sort(key=lambda x: x[0]) #chronological order
    for tri in range(len(tracks)-1):
        if tracks[tri][2]:
            di = 1
            tracks_overlap_check = tracks[tri][0]+tracks[tri][1] > tracks[tri+di][0]
            while tracks_overlap_check: #checks all possibilities
                tracks[tri+di][2] = False
                tracks_rem+=1
                di+=1
                if tri+di < len(tracks):
                    tracks_overlap_check = tracks[tri][0]+tracks[tri][1] > tracks[tri+di][0]
                else:
                    tracks_overlap_check = False

    #go trough again and check for discoveries
    for tri in range(len(tracks)):
        #if detected and not yet discovered, set to discovered
        if tracks[tri][2] and (not discovered[tracks[tri][4]][0]):
            discovered[tracks[tri][4]] = [True,tri]

    scheduler_return = {}
    scheduler_return['tracks'] = tracks
    scheduler_return['discovered'] = discovered
    scheduler_return['removed_tracks'] = tracks_rem
    return scheduler_return




def scheduling_movie(tracks, tracks_t, radar, population, root, time_slice=0.2, time_len = 1.0/60.0/30.0):
    #create a better sorted data structure
    tx = radar._tx[0]

    point_data = []
    track_data = []
    for track,t_track in zip(tracks,tracks_t):
        if track[2]:
            for t in t_track:
                #tracklet point time, OID, source
                row = [t, track[4], track[6]]
                point_data.append( row )
            #det time, det length, OID
            track_data.append([track[0], track[1], track[4]])

    #sort according to time, should be coherrent-int sec spaced
    point_data.sort(key=lambda row: row[0])
    track_data.sort(key=lambda row: row[0])

    point_data_arr = np.array([x[0] for x in point_data])

    if point_data[-1][0] - point_data[0][0] >= time_len*3600.0:
        t_lims = (point_data[0][0], point_data[0][0] + time_len*3600.0)
    else:
        t_lims = (point_data[0][0], point_data[-1][0])

    
    plt.style.use('dark_background')

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax, alpha = 0.2, color='white')

    ax.grid(False)
    plt.axis('off')

    def data_gen():
        t_curr = t_lims[0]
        data_list = {}
        trac_cnt = 0
        active_list = []
        t_ind = {}
        while t_curr < t_lims[1]:
            print(t_curr,' - ', t_lims[1])
            t_exec_y = time.time()

            t_curr += time_slice

            print('Adding tracks')
            check_tracks = True
            while check_tracks:
                track = track_data[trac_cnt]

                check_tracks = track_data[trac_cnt+1][0] <= t_curr
                print('Adding:', t_curr, trac_cnt, track[0], check_tracks, 'next - ', track_data[trac_cnt+1][0], ' df ', track[1]/time_slice)

                if track[0] <= t_curr and trac_cnt not in active_list:
                    t_vec = np.arange(t_curr, track[0] + track[1], time_slice)
                    if len(t_vec) > 0:
                        space_o = population.get_object(int(track[2]))
                        ecef_traj = space_o.get_orbit(t_vec)

                        #print(trac_cnt, track[0], track[1], t_vec)

                        data_list[trac_cnt] = ecef_traj
                        t_ind[trac_cnt] = 0
                        
                        active_list.append(trac_cnt)
                    
                    trac_cnt += 1

            print('removing tracks')
            del_lst = []
            for aci,tri in enumerate(active_list):
                track = track_data[tri]
                if track[0]+track[1] <= t_curr:

                    del data_list[tri]
                    del t_ind[tri]
                    del_lst.append(aci)

            del_lst.sort()
            del_lst = del_lst[::-1]

            for tri in del_lst:
                del active_list[tri]

            for ind,t_val in t_ind.items():
                t_ind[ind] = t_val + 1


            dt_arr = np.abs(point_data_arr - t_curr)

            point_ind = int(np.argmin(dt_arr))
            point_t = point_data[point_ind][0]
            point_oid = point_data[point_ind][1]

            space_o = population.get_object(int(point_oid))
            point = space_o.get_orbit(point_t)
            point.shape = (point.size,)

            print('yield time: ', (time.time() - t_exec_y)*60.0, ' min')
            

            yield t_ind, data_list, point, t_curr

    def run(data):
        # update the data
        t_ind, data_list, point, t_curr = data

        print((t_curr - t_lims[0])/(t_lims[1] - t_lims[0]),' done: ', 1.0/((t_curr - t_lims[0])/(t_lims[1] - t_lims[0]))*(time.time() - t0_exec)/3600.0, ' h left')

        titl.set_text('Simulation t=%.4f min' % (t_curr/60.0))

        print('Updating plot')
        for tri,dat in data_list.items():
            if len(ax_traj_list[tri].get_xydata()) == 0:
                print('- traj ', tri)
                ax_traj_list[tri].set_data(
                    dat[0,:],
                    dat[1,:],
                    )
                ax_traj_list[tri].set_3d_properties(
                    dat[2,:],
                    )
                ax_traj_list[tri].figure.canvas.draw()

            if t_ind[tri] < dat.shape[1]:
                print('- point ', tri)
                ax_point_list[tri].set_data(
                    dat[0,t_ind[tri]],
                    dat[1,t_ind[tri]],
                    )
                ax_point_list[tri].set_3d_properties(
                    dat[2,t_ind[tri]],
                    )
                ax_point_list[tri].figure.canvas.draw()

        for tri,tr_ax in enumerate(ax_traj_list):
            if len(tr_ax.get_xydata()) > 0:
                if tri not in data_list:
                    ax_traj_list[tri].set_data([],[])
                    ax_traj_list[tri].set_3d_properties([])
                    ax_traj_list[tri].figure.canvas.draw()

                    ax_point_list[tri].set_data([],[])
                    ax_point_list[tri].set_3d_properties([])
                    ax_point_list[tri].figure.canvas.draw()


        txp0,k0=tx.get_scan(t_curr).antenna_pointing(t_curr)

        print('- scan ')
        beam_len = 1000e3
        ax_tx_scan.set_data(
            [tx.ecef[0],tx.ecef[0] + k0[0]*beam_len],
            [tx.ecef[1],tx.ecef[1] + k0[1]*beam_len],
            )
        ax_tx_scan.set_3d_properties([tx.ecef[2],tx.ecef[2] + k0[2]*beam_len])
        ax_tx_scan.figure.canvas.draw()

        print('- track')
        #radar beams
        ax_txb.set_data(
            [tx.ecef[0],point[0]],
            [tx.ecef[1],point[1]],
            )
        ax_txb.set_3d_properties([tx.ecef[2],point[2]])
        ax_txb.figure.canvas.draw()
        for rxi in range(len(radar._rx)):
            print('- reciv ', rxi)
            ax_rxb_list[rxi].set_data(
                [radar._rx[rxi].ecef[0],point[0]],
                [radar._rx[rxi].ecef[1],point[1]],
                )
            ax_rxb_list[rxi].set_3d_properties([radar._rx[rxi].ecef[2],point[2]])
            ax_rxb_list[rxi].figure.canvas.draw()
        print('returning axis')
        return ax_traj_list, ax_txb, ax_rxb_list, ax_tx_scan

    #traj
    #ax_traj = ax.plot(ecef_traj[0,:],ecef_traj[1,:],ecef_traj[2,:],alpha=1,color="white")
    ax_traj_list = []
    ax_point_list = []
    for track in track_data:
        ax_traj, = ax.plot([],[],[],alpha=0.5,color="white")
        ax_traj_list.append(ax_traj)

        ax_point, = ax.plot([],[],[],'.',alpha=1,color="yellow")
        ax_point_list.append(ax_point)

    #init
    ecef_point = tx.ecef

    #radar beams
    ax_txb, = ax.plot(
        [tx.ecef[0],ecef_point[0]],
        [tx.ecef[1],ecef_point[1]],
        [tx.ecef[2],ecef_point[2]],
        alpha=1,color="green",
        )

    ax_tx_scan, = ax.plot(
        [tx.ecef[0],ecef_point[0]],
        [tx.ecef[1],ecef_point[1]],
        [tx.ecef[2],ecef_point[2]],
        alpha=1,color="yellow",
        )
    ax_rxb_list = []
    for rx in radar._rx:
        ax_rxb, = ax.plot(
            [rx.ecef[0],ecef_point[0]],
            [rx.ecef[1],ecef_point[1]],
            [rx.ecef[2],ecef_point[2]],
            alpha=1,color="green",
            )
        ax_rxb_list.append(ax_rxb)

    delta = 1500e3
    ax.set_xlim([tx.ecef[0] - delta,tx.ecef[0] + delta])
    ax.set_ylim([tx.ecef[1] - delta,tx.ecef[1] + delta]) 
    ax.set_zlim([tx.ecef[2] - delta,tx.ecef[2] + delta]) 

    titl = fig.text(0.5,0.94,'',size=22,horizontalalignment='center')
    
    t0_exec = time.time()

    print('setup done, starting anim')
    ani = animation.FuncAnimation(fig, run, data_gen,
        blit=False,
        #interval=1.0e3*time_slice,
        repeat=True,
        )

    print('Anim done, writing movie')

    fps = int(1.0/time_slice)
    if fps == 0:
        fps = 10

    Writer = animation.writers['ffmpeg']
    writer = Writer(metadata=dict(artist='Daniel Kastinen'),fps=fps)
    ani.save(root + 'scheduler_movie.mp4', writer=writer)

    print('showing plot')
    plt.show()

