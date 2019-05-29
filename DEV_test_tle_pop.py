import population_library as plib
import matplotlib.pyplot as plt
import numpy as np


pop = plib.tle_snapshot('data/uhf_test_data/tle-201801.txt', sgp4_propagation=True)

print('pop len:', len(pop))
print(pop.objs.dtype)

pop.print_row(0)

obj0 = pop.get_object(0)
print(obj0)

ecef = obj0.get_state(0)
print(ecef)


pop = plib.tle_snapshot('data/uhf_test_data/tle-201801.txt', sgp4_propagation=False)

print('pop len:', len(pop))
print(pop.objs.dtype)

pop.print_row(0)

obj1 = pop.get_object(0)
print(obj1)

ecef = obj1.get_state(0)
print(ecef)


x = obj0.get_state(np.linspace(0,3*3600, num=300, dtype=np.float))
y = obj1.get_state(np.linspace(0,3*3600, num=300, dtype=np.float))

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(15, 5)

ax.plot(x[0,:],x[1,:],x[2,:],"-r",alpha=0.5,label='SGP4')
ax.plot(y[0,:],y[1,:],y[2,:],"-b",alpha=0.5,label='Orekit')

ax.legend()
plt.title('TLE snapshot propagation comparison')

plt.show()