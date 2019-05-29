from glob import glob

sim_root = '/home/danielk/IRF/E3D_PA/DRACON_sim/example/ns_fence_full_sst_maint'

lst_objs = glob(sim_root+'/tracklets/*/')

#filter out only object names
#this works since string is ..../ID/
#then split will create [...., ID,'']
#so the next to last element [-2] is the id
objs = [ x.split('/')[-2] for x in lst_objs]

#first 4 are for factor ID
master_objs = [x[4:] for x in objs]
factor_objs = [x[:4] for x in objs]


#so to list the tracklets of folder 44
open_id = 44

tracklets = glob(lst_objs[open_id]+'*.tdm')

tracklet_data = [tr.split('-') for tr in tracklets]
#format: time, master ID (including factor), tx_rx index
tracklet_data = [(tr[1],tr[2],tr[3][:3]) for tr in tracklet_data]

print('Tracklets for folder {}:'.format(lst_objs[open_id]))
for i,tr in enumerate(tracklet_data):
    print('\n-- PATH: {}'.format(tracklets[i]) )
    print('Master ID: {}'.format(tr[1][4:]))
    print('Factor ID: {}'.format(tr[1][:4]) )
    print('TX: {} to RX: {}'.format(tr[2][0], tr[2][2]))
    print('Unix time: {}'.format(tr[0]) )

print('-------------------')

lst_priors = glob(sim_root+'/prior/*_init.oem')

#here we dont have trailing /
priors = [ x.split('/')[-1] for x in lst_priors]

#remove end of file name
prior_id = [ x.split('_')[0] for x in priors]

#first 4 are factor ID
master_prior = [x[4:] for x in prior_id]
factor_prior = [x[:4] for x in prior_id]


#so to open prior 44 in the list the path is
open_id = 44
print('Path to prior {}: {}'.format(open_id, lst_priors[open_id]))
print('With ID: {}'.format(prior_id[open_id]))
print('Master ID: {}, factor ID: {}'.format(master_prior[open_id], factor_prior[open_id]))