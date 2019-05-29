
import numpy as n

import TLE_tools as tle
import dpt_tools as dpt

import matplotlib.pyplot as plt

JD0 = dpt.date_to_jd(1980, 01, 01)
JD1 = dpt.date_to_jd(2019, 01, 01)

jd_v = n.linspace(JD0, JD1, num=1000, dtype=n.float)

data_PM = tle.get_Polar_Motion(jd_v)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(jd_v, n.degrees(data_PM[:,0]), label='$x_p$')
ax.plot(jd_v, n.degrees(data_PM[:,1]), label='$y_p$')
ax.set_xlabel('JD [days]')
ax.set_ylabel('Polar motion [deg]')
plt.legend()


fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(jd_v, n.sqrt((data_PM[:,0]*1e4)**2 + (data_PM[:,1]*1e4)**2))
ax.set_xlabel('JD [days]')
ax.set_ylabel('Shift [km]')
ax.set_title('approx Shift due to polar motion at 10,000 km range from Earth Center')


data_DUT = tle.get_DUT(jd_v)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(jd_v, data_DUT[:,0], label='UTC-UT1')
ax.set_xlabel('JD [days]')
ax.set_ylabel('DUT [s]')
plt.legend()


plt.show()