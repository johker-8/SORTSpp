import radar_library as rlib
import population_library as plib
import simulate_tracking as strack
import numpy as np


if __name__ == "__main__":
    
    # Here's what I've gots to do. With MPI, because this will take a lot of time
    # For each detectable object:
    # - calculate number of passes over 24 hours
    # - calculate mean time between passes over 24 hours (24.0/N_passes)
    # - estimate amount of time that it takes to decorrelate
    # - save info n_passes_per_day, max_spacing, mean_spacing, is_maintainable
    # plot results for population
    
    radar = rlib.eiscat_3d(beam='interp', stage=1)

    base = '/home/danielk/IRF/E3D_PA/FP_sims/'

    # space object population
    pop_e3d = plib.filtered_master_catalog_factor(
        radar = radar,
        detectability_file = base + 'celn_20090501_00_EISCAT_3D_PropagatorSGP4_10SNRdB.h5',
        treshhold = 0.01,
        min_inc = 50,
        prop_time = 48.0,
    )

    hour = 3600.0*24.0

    pop_e3d.delete(slice(None, 3))

    stop_t = 24.0*hour

    pass_dt = []
    pass_dt_div = []
    for space_o in pop_e3d.object_generator():
        print(space_o)
        pass_struct = strack.get_passes(space_o, radar, 0.0*hour, stop_t)

        plist = pass_struct["t"][0]
        dt = []
        for ind in range(len(plist)):
            if ind > 0:
                dt.append(plist[ind][0] - plist[ind-1][0])

        if len(plist) > 0:
            pass_dt_div.append(stop_t/float(len(plist)))
        else:
            pass_dt_div.append(np.nan)

        if len(dt) > 0:
            pass_dt.append(np.mean(dt))
        else:
            pass_dt.append(np.nan)

    pass_dt = np.array(pass_dt)
    pass_dt_div = np.array(pass_dt_div)

    print(pass_dt - pass_dt_div)

    print(pass_dt_div)
    print(pass_dt)
