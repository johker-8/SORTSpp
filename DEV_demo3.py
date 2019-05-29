#project 3: Lets see if our spiral scan and Aland radar can detect an object in space

from DEV_demo1 import SC
from DEV_demo2 import radar
import simulate_scan
import space_object

import matplotlib.pyplot as plt

#OOPS we need to set the scan to the radar

radar.set_scan(SC)

#We need a space object
obj = space_object.SpaceObject(
    a=7000, e=0.0, i=69,
    raan=0, aop=0, mu0=0,
    C_D=2.3, A=1.0, m=1.0,
    C_R=1.0, oid=42, d=1.0,
    mjd0=57125.7729,
)

#scan for 24 h
detections = simulate_scan.get_detections(obj, radar, 0.0, 3600.0*24.0)


#lets look at the detections manually
simulate_scan.pp_det(detections)

#We can visualize the detections
#simulate_scan.plot_scan_for_object(obj, radar, 0.0, 3600.0*24.0, plot_full_scan=True)

#plt.show()