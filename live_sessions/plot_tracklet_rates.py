# coding: utf-8
h = h5py.File(root + 'scheduled_maint_dynamic_sceduler_q22_ti1.h5','r')

names = h.keys()
print(names)

data = h[names[3]].value
fig = figure()
ax = fig.add_subplot(1,1,1)

data_len = n.max(data) - n.min(data)

hst = ax.hist(data,int(n.round(data_len)))
ax.set_ylabel('Tracklets per hour', fontsize=23)
ax.set_xlabel('Simulation time [h]', fontsize=23)
ax.set_title('Scheduled tracklets frequency: Full catalogue maintenance', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

data = h[names[4]].value
data = data*60.0/0.2
fig = figure()
ax = fig.add_subplot(1,1,1)
hst = ax.hist(data,n.arange(-0.5,12.5,1))
ax.set_ylabel('Tracklets', fontsize=23)
ax.set_xlabel('Number of tracklet points', fontsize=23)
ax.set_title('Scheduled track time: Full catalogue maintenance', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
