# coding: utf-8
h = open_meta(0)
names = h.keys()
print(names)
data = h[names[3]].value
fig = figure()
ax = fig.add_subplot(1,1,1)
hst = ax.hist(data, 24*2-1, cumulative=True, label='Detected objects')
ax.set_ylabel('Objects', fontsize=23)
ax.set_xlabel('Time [h]', fontsize=23)
line = ax.plot([0,48],[73886,73886],'-r',label='Detectable population')
ax.set_title('Cold start catalogue build-up: North-South random fence scan', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
