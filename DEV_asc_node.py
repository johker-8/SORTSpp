import numpy as np
import scipy as sp

from matplotlib import pyplot as plt

import dpt_tools as dpt
import orbit_verification as over

sv, _, _ = over.read_poe('data/S1A_OPER_AUX_POEORB_OPOD_20150527T122640_V20150505T225944_20150507T005944.EOF')

# sv = sv[:177]
sv = sv[177:750]

taan = np.asarray([dpt.ascending_node_from_statevector(s, 2300) for s in sv])
svt = sv.tai

plt.gcf(); plt.clf()
plt.plot((svt-svt[0])/dpt.sec, taan/dpt.sec)
plt.xlabel('Time since last ascending node')
plt.ylabel('Time since last AN estimation')
plt.show()

plt.gcf(); plt.clf()
plt.plot((svt-svt[0])/dpt.sec, (svt-svt[0])/dpt.sec-taan/dpt.sec)
plt.xlabel('Time since last ascending node')
plt.ylabel('Time since last AN estimation error')
plt.show()