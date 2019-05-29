import os

import pymc3 as pm
import scipy
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as tt

import propagator_sgp4

prop = propagator_sgp4.PropagatorSGP4(
    polar_motion = False,
    out_frame = 'TEME',
)

# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called.
    """
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, likelihood, r, sigma_r, station, dt, mjd0):
        """
        Initialise the Op
        """

        # add inputs as class attributes
        self.likelihood = likelihood
        self.r = r
        self.sigma_r = sigma_r
        self.station = station
        self.dt = dt
        self.mjd0 = mjd0

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        state, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(state, self.r, self.sigma_r, self.station, self.dt, self.mjd0)

        outputs[0][0] = np.array(logl) # output the log-likelihood

def r_likelihood(state, r, sigma_r, station, dt, mjd0):

    state_prop = prop.get_orbit_cart(
        t = dt,
        x = state[0],
        y = state[1],
        z = state[2],
        vx = state[3],
        vy = state[4],
        vz = state[5],
        mjd0 = mjd0,
        B = 0.0,
    )
    
    r_sim = np.linalg.norm(station - state_prop[:3,0])
    return scipy.stats.norm.logpdf(r_sim, r, sigma_r)


def determine_orbit(prior, r, sigma_r, station, dt, mjd0):
    
    logl = LogLike(r_likelihood, r, sigma_r, station, dt, mjd0)
    
    with pm.Model() as model:
    
        # Priors for unknown model parameters
        state = pm.MvNormal('state', mu=prior['mu'], cov=prior['cov'], shape=(6,))
        
        theta = tt.as_tensor_variable(state)
        observed = {'theta': theta}
        
        exp_surv = pm.DensityDist('like', lambda theta: logl(theta), observed=observed)
    
        trace = pm.sample(100)
        
        pm.traceplot(trace)
    
    plt.show()
    return trace


if __name__ == '__main__':

    state0 = [2721793.785377, 1103261.736653, 6427506.515945, 6996.001258,  -171.659563, -2926.43233]
    state0 = np.array(state0)
    mjd0 = 57125.7729
    t = 3500.0
    
    prior = {}
    prior['cov'] = np.diag([1e3, 1e3, 1e3, 1e1, 1e1, 1e1])
    prior['mu'] = state0
    
    x, y, z, vx, vy, vz = state0

    ecefs = prop.get_orbit_cart(t, x, y, z, vx, vy, vz, mjd0=mjd0, B = 0.0)
    
    station = np.array([0,0,6300e3])
    
    sigma_r = 10e3
    r = np.random.randn(1)[0]*sigma_r + np.linalg.norm(ecefs[:3,0] - station) + 200.0
    
    
    trace = determine_orbit(prior, r, sigma_r, station, t, mjd0 = mjd0)