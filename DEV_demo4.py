#Project 4: Lets see how much of MASTER 2009 this carzy radar can see

import population_library
import population_filter

from DEV_demo1 import SC
from DEV_demo2 import radar
radar.set_scan(SC)

pop = population_library.master_catalog()

pop.delete(slice(100, None)) #OOOPS! Lets only keep the first 100 objects

#Todo: add return parameters to documentation :)
detectable, n_rx_dets, peak_snrs = population_filter.filter_objects(
    radar, 
    pop, 
    ofname=None, 
    prop_time=24.0,
)