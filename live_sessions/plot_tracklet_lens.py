# coding: utf-8
h = open_meta(0)
names = h.keys()
print(names)
data = h[names[4]].value
data = data*60.0/0.2
data = data[data < 10]
fig = figure()
ax = fig.add_subplot(1,1,1)
hst = ax.hist(data)
ax.set_ylabel('Objects', fontsize=23)
ax.set_xlabel('Number of tracklet points', fontsize=23)
ax.set_title('Discovered object track time: North-South random fence scan', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
