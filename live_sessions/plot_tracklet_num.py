# coding: utf-8
#h = open_meta(1)
h = h5py.File(root + 'scheduled_tracks_dynamic_sceduler_q22_ti1.h5','r')

names = h.keys()
print(names)

data = h[names[0]].value
data = data*60.0
fig = figure()
ax = fig.add_subplot(1,1,1)
hst = ax.hist(data,100)
ax.set_ylabel('Tracklets', fontsize=23)
ax.set_xlabel('Tracklet mean displacement from peark SNR [s]', fontsize=23)
ax.set_title('Scheduled tracklets point distribution: Full catalogue maintenance', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

data = h[names[8]].value
data = data*60.0
fig = figure()
ax = fig.add_subplot(1,1,1)
hst = ax.hist(data,16)
ax.set_ylabel('Tracklets', fontsize=23)
ax.set_xlabel('Tracklet span [s]', fontsize=23)
ax.set_title('Scheduled tracklets points reach: Full catalogue maintenance', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

