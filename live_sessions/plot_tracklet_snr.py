# coding: utf-8
h = open_meta(1)
names = h.keys()
print(names)
data = h[names[6]].value
fig = figure()
ax = fig.add_subplot(1,1,1)
hst = ax.hist(data)
ax.set_ylabel('Tracklets', fontsize=23)
ax.set_xlabel('Tracklet max SNR [dB]', fontsize=23)
ax.set_title('Discovered object follow-up SNR: North-South random fence scan', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
