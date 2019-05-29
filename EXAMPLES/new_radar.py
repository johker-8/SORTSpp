import antenna_library as a
import radar_scan_library as rslib
import radar_config as rc

def eiscat_uhf():
   uhf_lat=69.34023844
   uhf_lon=20.313166
   uhf=rc.rx_antenna("UHF Tromso",
               uhf_lat,
               uhf_lon,
               30,
               930e6,
               100,
               a.cassegrain_beam(az0=0,el0=90,lat=uhf_lat,lon=uhf_lon,I_0=10**4.3,f=930e6,a0=16.0,a1=4.58/2.0))


   uhf_tx=rc.tx_antenna("UHF Tromso TX",
                  uhf_lat,
                  uhf_lon,
                  30,
                  930e6,
                  100,
                  a.cassegrain_beam(0,90,uhf_lat,uhf_lon,I_0=10**4.3,a0=16.0,a1=4.58/2.0,f=930e6),
                  rslib.beampark_model(az = 90.0, el = 75.0, lat = uhf_lat,lon = uhf_lon,alt = 0.0),
                  2e6,  # 2 MW
                  1e6,  # 1 MHz
                  0.125) # 12.5% duty-cycle

   # EISCAT UHF beampark
   tx=[uhf_tx]
   rx=[uhf]

   euhf = rc.radar_system(tx,rx,'Eiscat UHF')
   return euhf

