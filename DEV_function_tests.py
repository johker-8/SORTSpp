import numpy as np

import keplerian_sgp4 as ksgp4

def gstime_test(verbose=True):
	print('########## gstime_test\n\n')

	gstime_ref = 5.45956256715196
	jdut1 = 2453101.82740678

	gstime = ksgp4.gstime(jdut1)
	
	if verbose:
		print('gs time test\n')
		print('jdut1=',jdut1,' \n')
		print('gstime_ref=\n', gstime_ref )
		print('gstime=\n', gstime )

	TEST = np.abs(gstime - gstime_ref) < 1e-5

	if TEST:
		print('polarm_test passed [OK]')
	else:
		print('polarm_test failed [ERROR]')
	return TEST


def polarm_test(verbose=True):
	print('########## polarm_test\n\n')

	pm_ref = np.matrix('0.999999999999767,0,6.82045582858465e-07; \
       -1.102136303876e-12,0.999999999998694,1.6159276323683e-06; \
     -6.82045582857574e-07,-1.61592763236868e-06,0.999999999998462')

	xv = np.matrix([[1.0],[1.0],[1.0]])

	conv = np.pi / (180.0*3600.0)
	opt = '20'
	xp   = -0.140682 * conv
	yp   =  0.333309 * conv
	ttt = 0.0426236318889942

	pm = ksgp4.polarm(xp, yp, ttt, opt)

	if verbose:
		print('polar motion matrix tests\n')
		print('xp=',xp,',yp=',yp,'ttt=',ttt,'opt=',opt,' \n')
		print('pm=\n', pm )
		print('pm=\n', pm.T )

	xvt = np.linalg.inv(pm)*(pm*xv)

	TEST = np.allclose(pm, pm_ref) and np.allclose(xvt, xv)

	if TEST:
		print('polarm_test passed [OK]')
	else:
		print('polarm_test failed [ERROR]')
	return TEST



def teme2ecef_test(verbose=True):
	print('########## teme2ecef_test \n\n')

#	yv_ref = np.matrix('-1033.479383,-1033.479383;\
#						7901.2952754,7901.2952754;\
#						6380.3565958,6380.3565958')
	yv_ref = np.matrix('1033.479383,1033.479383;\
						-7901.2952754,-7901.2952754;\
						6380.3565958,6380.3565958')

	xv = np.matrix('5094.18010717893,5094.18010717893; \
					6127.64470516667,6127.64470516667; \
					6380.34453274887,6380.34453274887')
	t = np.matrix([0,0])

	mjd0 = 2453101.82740678 - 2400000.5
	ttt = (mjd0 - 51544.5)/(365.25*100.0)
	yv = ksgp4.teme2ecef(t,xv,mjd0)
	if verbose:
		print('xv =\n',xv,' \n')
		print('t =',t,'mjd0=',mjd0,' \n')
		print('ttt =',ttt,' \n')
		print('yv =\n',yv,' \n')
		print('yv_ref =\n',yv_ref,' \n')

	TEST = np.allclose(yv, yv_ref) and yv.shape[1]==xv.shape[1] and yv.shape[0]==xv.shape[0]
	#print('shapes: ',yv.shape,',',xv.shape)
	print(yv-yv_ref)

	if TEST:
		print('teme2ecef_test passed [OK]')
	else:
		print('teme2ecef_test failed [ERROR]')
	return TEST




if __name__ == "__main__":
	passed = 0
	tests = 0
	verbose = True

	tests+=1
	if polarm_test(verbose):
		passed+=1
	tests+=1
	if teme2ecef_test(verbose):
		passed+=1
	tests+=1
	if gstime_test(verbose):
		passed+=1

	print('\n ####################### \n \n',\
		passed,' tests of ',tests, ' passed\n')
