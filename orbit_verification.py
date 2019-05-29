
import re
import numpy as np

import dpt_tools as dpt

# Descriptor for numpy dtype representing a state vector
svec_d = [
   ('utc',        'datetime64[ns]'),          # absolute time in UTC
   ('tai',        'datetime64[ns]'),          # absolute time in TAI
   ('pos',         np.double, (3,)),          # ECEF position [m]
   ('vel',         np.double, (3,))]          # ECEF velocity [m/s]

def read_poe(fname):
    utc_r = re.compile(r'<(UTC|TAI)>\1=([0-9T:.-]+)</\1>')
    tag_r = re.compile(r'<([A-Za-z_]+)>(.+)</\1>')
    xyz_r = re.compile(r'<(V?[XYZ]) unit="m(/s)?">([+-]?\d+\.\d+)</\1>')

    sv_data = []
    current = None
    for line in open(fname):
        if '<OSV>' in line:
            current = {}
        elif '</OSV>' in line:
            newitem = {}
            newitem['pos'] = [current.pop('X', None),
                              current.pop('Y', None),
                              current.pop('Z', None)]
            newitem['vel'] = [current.pop('VX', None),
                              current.pop('VY', None),
                              current.pop('VZ', None)]
            newitem['tai'] = current.pop('TAI', None)
            newitem['utc'] = current.pop('UTC', None)
            newitem['abs_orbit'] = current.pop('Absolute_Orbit', None)
            newitem['quality'] = current.pop('Quality', None)
            sv_data.append(newitem)
            current = None
        else:
            if current is None:
                continue
            try:
                m = utc_r.search(line)
                if m:
                    g = m.groups()
                    current[g[0]] = g[1]
                    continue
                m = xyz_r.search(line)
                if m:
                    g = m.groups()
                    current[g[0]] = float(g[2])
                    continue
                m = tag_r.search(line)
                if m:
                    g = m.groups()
                    if g[0] == 'Absolute_Orbit':
                        current[g[0]] = int(g[1])
                    else:
                        current[g[0]] = g[1]  # qual[m.groups()[1]]
            except TypeError:
                print("Trying add fields to abandoned statevector")
            except ValueError:
                print("Line <{0}> not understood".format(line))

    sv = np.recarray(len(sv_data), dtype=svec_d)
    sv['tai'] = [s['tai'] for s in sv_data]
    sv['utc'] = [s['utc'] for s in sv_data]
    sv['pos'] = [s['pos'] for s in sv_data]
    sv['vel'] = [s['vel'] for s in sv_data]

    abs_orbit = [s['abs_orbit'] for s in sv_data]
    quality = [s['quality'] for s in sv_data]
    return sv, abs_orbit, quality

