#!/usr/bin/env python

'''Orbit determination module.

'''

#
#    Standard python
#
import os
import copy
import glob

#
#   External packages
#
import h5py
from mpi4py import MPI
from pandas.plotting import scatter_matrix
import pandas
from tqdm import tqdm
import scipy
import scipy.stats
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.path as mpath

comm = MPI.COMM_WORLD

#
#   SORTS ++
#
import space_object
import dpt_tools as dpt
import TLE_tools as tle
import ccsds_write
import plothelp
import sources

from sorts_config import p as default_propagator
default_propagator_inst = default_propagator()


def _named_to_enumerated(state, names):
    return np.array([state[name] for name in names], dtype=np.float64).flatten()
    

def _enumerated_to_named(state, names):
    _dtype = [(name, 'float64') for name in names]
    _state = np.empty((1,), dtype=_dtype)
    for ind, name in enumerate(names):
        _state[name] = state[ind]
    return _state



def mpi_wrap(run):

    def new_run(self, *args, **kwargs):

        run_mpi = self.kwargs.get('MPI', True)
        if run_mpi:
            steps = self.kwargs['steps']
            self.kwargs['steps'] = len(range(comm.rank, steps, comm.size))

            results0 = run(self, *args, **kwargs)
            trace0 = results0.trace

            trace = np.empty((0,), dtype=trace0.dtype)
            for pid in range(comm.size):
                trace_pid = comm.bcast(trace0, root=pid)
                trace = np.append(trace, trace_pid)

            self.results.trace = trace
            self._fill_results()

            return self.results
        else:
            return run(self, *args, **kwargs)
        

    return new_run



def mpi_wrap_sets(run):

    def new_run(self, *args, **kwargs):

        raise NotImplementedError()
        steps = self.kwargs['steps']
        self.kwargs['steps'] = len(range(comm.rank, steps, comm.size))

        results0 = run(self, *args, **kwargs)
        trace0 = results0.trace

        trace = np.empty((0,), dtype=trace0.dtype)
        for pid in range(comm.size):
            trace_pid = comm.bcast(trace0, root=pid)
            trace = np.append(trace, trace_pid)

        self.results.trace = trace
        self._fill_results()

        return self.results

    return new_run





class ForwardModel(object):

    dtype = [] #this is the dtype that is returned by the model

    REQUIRED_DATA = [
        'date',
        'date0',
        'params',
    ]

    def __init__(self, data, propagator, coord='cart', **kwargs):
        for key in self.REQUIRED_DATA:
            if key not in data:
                raise ValueError('Data field {} is mandatory for {}'.format(key, type(self).__name__))
        
        self.data = data
        self.propagator = propagator
        self.coord = coord
        
        self.data['mjd0'] = dpt.npdt2mjd(self.data['date0'])
        t = (self.data['date'] - self.data['date0'])/np.timedelta64(1, 's')
        self.data['t'] = t

    def get_states(self, state):

        args = self.data['params'].copy()

        for key in state.dtype.names:
            args[key] = state[0][key]

        if self.coord == 'cart':
            states = self.propagator.get_orbit_cart(
                t = self.data['t'],
                mjd0 = self.data['mjd0'],
                **args
            )
        elif self.coord == 'kep':
            states = self.propagator.get_orbit(
                t = self.data['t'],
                mjd0 = self.data['mjd0'],
                **args
            )
        else:
            raise ValueError('Coordinate system not recognized: {}'.format(self.coord))

        return states

    def evaluate(self, state):
        '''Evaluate forward model
        '''
        raise NotImplementedError()



class RadarPair(ForwardModel):

    dtype = [
        ('r', 'float64'),
        ('v', 'float64'),
    ]
    
    REQUIRED_DATA = ForwardModel.REQUIRED_DATA + [
        'tx_ecef',
        'rx_ecef',
    ]

    def __init__(self, data, propagator, coord='cart', **kwargs):
        super(RadarPair, self).__init__(data, propagator, coord, **kwargs)


    @staticmethod
    def generate_measurements(state_ecef, rx_ecef, tx_ecef):

        r_tx = tx_ecef - state_ecef[:3]
        r_rx = rx_ecef - state_ecef[:3]

        r_tx_n = np.linalg.norm(r_tx)
        r_rx_n = np.linalg.norm(r_rx)
        
        r_sim = r_tx_n + r_rx_n
        
        v_tx = -np.dot(r_tx, state_ecef[3:])/r_tx_n
        v_rx = -np.dot(r_rx, state_ecef[3:])/r_rx_n

        v_sim = v_tx + v_rx

        #print(state_ecef)
        #print(r_tx_n, r_rx_n)
        #print(rx_ecef, tx_ecef)

        return r_sim, v_sim


    def evaluate(self, state):
        '''Evaluate forward model
        '''

        states = self.get_states(state)

        obs_dat = np.empty((len(self.data['t']), ), dtype=RadarPair.dtype)

        for ind in range(len(self.data['t'])):
            r_obs, v_obs = RadarPair.generate_measurements(states[:,ind], self.data['rx_ecef'], self.data['tx_ecef'])
            obs_dat[ind]['r'] = r_obs
            obs_dat[ind]['v'] = v_obs

        return obs_dat



class EstimatedState(ForwardModel):

    dtype = [
        ('x', 'float64'),
        ('y', 'float64'),
        ('z', 'float64'),
        ('vx', 'float64'),
        ('vy', 'float64'),
        ('vz', 'float64'),
    ]

    def __init__(self, data, propagator, coord='cart', **kwargs):
        super(EstimatedState, self).__init__(data, propagator, coord, **kwargs)


    def evaluate(self, state):
        '''Evaluate forward model
        '''

        states = self.get_states(state)

        obs_dat = np.empty((len(self.data['t']), ), dtype=EstimatedState.dtype)

        for ind in range(len(self.data['t'])):
            for dim, npd in enumerate(EstimatedState.dtype):
                name, _ = npd
                obs_dat[ind][name] = states[dim,ind]

        return obs_dat



class Parameters(object):

    attrs = [
        'variables',
        'trace',
        'MAP',
        'residuals',
        'date',
    ]

    def __init__(self, **kwargs):
        for key in self.attrs:
            setattr(self, key, kwargs.get(key, None))

    @classmethod
    def load_h5(cls, path):
        '''Load evaluated posterior
        '''
        results = cls()
        if isinstance(path, sources.Path):
            if path.ptype != 'file':
                raise TypeError('Can only load posterior data from file, not "{}"'.format(path.ptype))
            _path = path.data
        else:
            _path = path

        with h5py.File(_path, 'r') as hf:
            results.variables = hf.attrs['variables'].tolist()
            results.date = dpt.mjd2npdt(hf.attrs['date'])
            results.trace = hf['trace'].value
            results.MAP = hf['MAP'].value
            results.residuals = []
            grp = hf['residuals/']
            for key in grp:
                results.residuals.append(grp[key].value)

        return results


    def __getitem__(self, key):
        if key in self.trace.dtype.names:
            return self.trace[key]
        else:
            return KeyError('No results exists for "{}"'.format(key))


    def load(self, path):
        '''Load evaluated posterior
        '''
        if isinstance(path, sources.Path):
            if path.ptype != 'file':
                raise TypeError('Can only load posterior data from file, not "{}"'.format(path.ptype))
            _path = path.data
        else:
            _path = path

        with h5py.File(_path, 'r') as hf:
            _vars = hf.attrs['variables'].tolist()
            if self.variables is not None:
                for var in _vars:
                    if var not in self.variables:
                        raise Exception('Variable spaces do not match between current and loaded data')
                for var in self.variables:
                    if var not in _vars:
                        raise Exception('Variable spaces do not match between current and loaded data')
            else:
                self.variables = _vars
            if self.results is not None:
                self.results = np.append(self.results, hf['results'].value)
            else:
                self.results = hf['results']
            if self.MAP is not None:
                self.MAP = np.append(self.MAP, hf['MAP'].value)
            else:
                self.MAP = hf['MAP']
            if self.residuals is not None:
                grp = hf['residuals/']
                for key in grp:
                    results.residuals.append(grp['{}'.format(ind)].value)
            else:
                self.residuals = []
                grp = hf['residuals/']
                for key in grp:
                    self.residuals.append(grp['{}'.format(ind)].value)
            if self.date is not None:
                if self.date != dpt.mjd2npdt(hf.attrs['date']):
                    raise Exception('Cannot load data from another epoch "{}" vs "{}"'.format(self.date, dpt.mjd2npdt(hf.attrs['date'])))



    def save(self, path):
        '''Save evaluated posterior
        '''
        if isinstance(path, sources.Path):
            if path.ptype != 'file':
                raise TypeError('Can only write posterior data to file, not "{}"'.format(path.ptype))
            _path = path.data
        else:
            _path = path

        with h5py.File(_path, 'w') as hf:
            hf.attrs['variables'] = self.variables
            hf['trace'] = self.trace
            hf['MAP'] = self.MAP
            hf.attrs['date'] = dpt.npdt2mjd(self.date)

            grp = hf.create_group("residuals")
            for ind, resid in enumerate(self.residuals):
                grp.create_dataset(
                        '{}'.format(ind),
                        data=self.residuals[ind],
                    )


    def autocovariance(self, max_k = None, min_k = None):
        if max_k is None:
            max_k = len(self.trace)
        else:
            if max_k >= len(self.trace):
                max_k = len(self.trace)

        if min_k is None:
            min_k = 0
        else:
            if min_k >= len(self.trace):
                min_k = len(self.trace)-1


        gamma = np.empty((max_k-min_k,), dtype=self.trace.dtype)

        _n = len(self.trace)

        for var in self.variables:
            for k in range(min_k, max_k):
                covi = self.trace[var][:(_n-k)] - self.MAP[0][var]
                covik = self.trace[var][k:_n] - self.MAP[0][var]
                gamma[var][k] = np.sum( covi*covik )/float(_n)

        return gamma


    def batch_mean(self, batch_size):
        if batch_size > len(self.trace):
            raise Exception('Not enough samples to calculate batch statistics')

        _max = batch_size
        batches = len(self.trace)//batch_size
        batch_mean = np.empty((batches,), dtype=self.trace.dtype)
        for ind in range(batches):
            batch = self.trace[(_max - batch_size):_max]
            _max += batch_size

            for var in self.variables:
                batch_mean[ind][var] = np.mean(batch[var])

        return batch_mean


    def batch_covariance(self, batch_size):
        if batch_size > len(self.trace):
            raise Exception('Not enough samples to calculate batch statistics')

        batch_mean = self.batch_mean(batch_size)

        _max_str = int(np.max([len(var) for var in self.variables]))

        _dtype = self.trace.dtype.names
        _dtype = [('variable', 'U{}'.format(_max_str))] + [(name, 'float64') for name in _dtype]
        cov = np.empty((len(self.variables),), dtype=_dtype)
        for ind, xvar in enumerate(self.variables):
            for yvar in self.variables:
                cov[ind]['variable'] = xvar
                cov[ind][yvar] = np.mean( (batch_mean[xvar] - self.MAP[xvar])*(batch_mean[yvar] - self.MAP[yvar]) )/float(len(batch_mean))

        return cov


    def batch_variance(self, batch_size):
        if batch_size > len(self.trace):
            raise Exception('Not enough samples to calculate batch statistics')

        batch_mean = self.batch_mean(batch_size)

        variance = np.empty((1,), dtype=self.trace.dtype)
        for var in self.variables:
            variance[var] = np.mean( (batch_mean[var] - self.MAP[var])**2)

        return variance/float(len(batch_mean))


    def covariance_mat(self, variables=None):
        if variables is None:
            variables = self.variables

        cov = np.empty((len(variables),len(variables)), dtype=np.float64)

        mean = np.empty((1,), dtype=self.trace.dtype)
        for ind, xvar in enumerate(variables):
            mean[xvar] = np.mean(self.trace[xvar])

        for xind, xvar in enumerate(variables):
            for yind, yvar in enumerate(variables):
                cov[xind, yind] = np.sum( (self.trace[xvar] - mean[xvar])*(self.trace[yvar] - mean[yvar]) )/float(len(self.trace)-1)
        return cov


    def covariance(self):

        _max_str = int(np.max([len(var) for var in self.variables]))

        _dtype = self.trace.dtype.names
        _dtype = [('variable', 'U{}'.format(_max_str))] + [(name, 'float64') for name in _dtype]
        cov = np.empty((len(self.variables),), dtype=_dtype)

        mean = np.empty((1,), dtype=self.trace.dtype)
        for ind, xvar in enumerate(self.variables):
            mean[xvar] = np.mean(self.trace[xvar])

        for ind, xvar in enumerate(self.variables):
            for yvar in self.variables:
                cov[ind]['variable'] = xvar
                cov[ind][yvar] = np.sum( (self.trace[xvar] - mean[xvar])*(self.trace[yvar] - mean[yvar]) )/float(len(self.trace)-1)
        return cov

    def __str__(self):
        _str = ''
        _str += '='*10 + ' MAP state ' + '='*10 + '\n'
        for ind, var in enumerate(self.variables):
            _str += '{}: {}\n'.format(var, self.MAP[var])

        _str += '='*10 + ' Residuals ' + '='*10 + '\n'
        for ind, res in enumerate(self.residuals):
            _str += ' - Model {}'.format(ind) + '\n'
            for key in res.dtype.names:
                _str += ' -- mean({}) = {}'.format(key, np.mean(res[key])) + '\n'
        return _str



class Posterior(object):

    REQUIRED_DATA = []
    REQUIRED = []
    OPTIONAL = {}

    def __init__(self, data, variables, **kwargs):
        self.kwargs = self.OPTIONAL.copy()
        self.kwargs.update(kwargs)

        self.variables = variables
        self.data = data

        self.results = Parameters(variables = variables)


    def logprior(self, state):
        '''The logprior function, defaults to uniform if not implemented
        '''
        return 0.0


    def loglikelihood(self, state):
        '''The loglikelihood function
        '''
        raise NotImplementedError()

    
    def evalute(self, state):
        return self.logprior(state) + self.loglikelihood(state)


    def run(self):
        '''Evaluate posterior
        '''
        raise NotImplementedError()


class OptimizeLeastSquares(Posterior):

    REQUIRED_DATA = [
        'sources',
        'Model',
        'date0',
        'params',
    ]

    REQUIRED = [
        'start',
        'propagator',
    ]

    OPTIONAL = {
        'method': 'Nelder-Mead',
        'prior': None,
        'options': {},
        'coord': 'cart',
    }

    def __init__(self, data, variables, **kwargs):
        super(OptimizeLeastSquares, self).__init__(data, variables, **kwargs)

        for key in self.REQUIRED:
            if key not in kwargs:
                raise ValueError('Argument "{}" is mandatory'.format(key))

        for key in self.REQUIRED_DATA:
            if key not in data:
                raise ValueError('Data field "{}" is mandatory'.format(key))

        self._models = []

        for source in data['sources']:
            if not isinstance(source, sources.ObservationSource):
                raise ValueError('Non-observation data detected, "{}" not supported'.format(type(source)))

            req_args = copy.copy(self.data['Model'].REQUIRED_DATA)
            for arg in source.avalible_data():
                if arg in req_args:
                    req_args.remove(arg)

            model_data = {
                'propagator': self.kwargs['propagator'],
                'coord': self.kwargs['coord'],
            }
            for arg in req_args:
                if arg not in self.data:
                    raise TypeError('Model REQUIRED data "{}" not found'.format(arg))
                model_data[arg] = self.data[arg]

            model = source.generate_model(
                Model=self.data['Model'],
                **model_data
            )
            self._models.append(model)
        
        self._tmp_residulas = []
        for ind in range(len(self._models)):
            self._tmp_residulas.append(
                np.empty((len(self.data['sources'][ind].data),), dtype=self._models[ind].dtype)
            )
        

    def logprior(self, state):
        '''The logprior function
        '''
        logprob = 0.0
        
        if self.kwargs['prior'] is None:
            return logprob
        
        for prior in self.kwargs['prior']:
            _state = _named_to_enumerated(state, prior['variables'])
            dist = getattr(scipy.stats, prior['distribution'])
            
            _pr = dist.logpdf(_state, **prior['params'])
            if isinstance(_pr, np.ndarray):
                _pr = _pr[0]
            logprob += _pr
        
        return logprob



    def loglikelihood(self, state):
        '''The loglikelihood function
        '''

        tracklets = self.data['sources']
        n_tracklets = len(tracklets)

        logsum = 0.0
        for ind in range(n_tracklets):
            
            sim_data = self._models[ind].evaluate(state)

            for name, nptype in self._models[ind].dtype:
                _residuals = tracklets[ind].data[name] - sim_data[name]

                self._tmp_residulas[ind][name] = _residuals
                
                logsum -= np.sum(_residuals**2.0/(tracklets[ind].data[name + '_sd']**2.0))
        return 0.5*logsum


    def run(self):
        if self.kwargs['start'] is None and self.kwargs['prior'] is None:
            raise ValueError('No start value or prior given.')

        start = _named_to_enumerated(self.kwargs['start'], self.variables)

        maxiter = self.kwargs['options'].get('maxiter', 3000)
        max_inf = maxiter//10
        inf_c = 0

        def fun(x):
            global inf_c
            _x = _enumerated_to_named(x, self.variables)

            try:
                val = self.evalute(_x)
            except:
                #print(_x)
                #raise
                val = np.inf
                inf_c += 1
            else:
                if np.isnan(val):
                    inf_c += 1
                else:
                    inf_c = 0

            if inf_c > max_inf:
                raise Exception('Too many inf evaluations, something probably wrong...')

            val = -val
            
            pbar.update(1)
            pbar.set_description("Least Squares = {:<10.3f} ".format(val))

            return val
        
        print('\n{} running {}'.format(type(self).__name__, self.kwargs['method']))

        pbar = tqdm(total=maxiter, ncols=100)
        xhat = optimize.minimize(
            fun,
            start,
            method = self.kwargs['method'],
            options = self.kwargs['options'],
        )
        pbar.close()

        self.results.trace = _enumerated_to_named(xhat.x, self.variables)
        self.results.MAP = self.results.trace.copy()
        self.loglikelihood(self.results.MAP)
        self.results.residuals = copy.deepcopy(self._tmp_residulas)
        self.results.date = self.data['date0']

        return self.results


    def residuals(self, state):

        self.loglikelihood(state)

        residuals = []
        for ind, resid in enumerate(self._tmp_residulas):
            residuals.append({
                'date': self._models[ind].data['date'],
                'residuals': resid.copy(),
            })
        return residuals




def hamiltoniuan_monte_carlo(state, potential_field, grad_potential_field, epsilon, iters):
    state_new = state.copy()

    variables = list(enumerate(state.dtype.names))
    dim = len(variables)

    momentum = np.random.randn(dim)

    momentum_new = momentum.copy()

    grad = grad_potential_field(state_new)
    for ind, var in variables:
        momentum_new[ind] -= epsilon*grad[var]*0.5 #half step

    #concatenate steps to full steps according to Hamiltonian split mechanics
    for ind in range(iters-1):
        for ind, var in variables:
            state_new[var] += epsilon*momentum_new[ind]

        grad = grad_potential_field(state_new)
        for ind, var in variables:
            momentum_new[ind] -= epsilon*grad[var]

    for ind, var in variables:
        state_new[var] -= epsilon*momentum_new[ind] #last step

    grad = grad_potential_field(state_new)
    for ind, var in variables:
        momentum_new[ind] -= epsilon*grad[var]*0.5 #half step

    #now operator is reversible if we reverse momentum
    momentum_new = -1.0*momentum_new

    E_k = np.sum(momentum**2.0)*0.5
    E_k_new = np.sum(momentum_new)**2.0*0.5

    E_p = potential_field(state)
    E_p_new = potential_field(state_new)
    
    beta = np.exp(E_p - E_p_new + E_k - E_k_new)

    if np.random.rand(1) < beta:
        return state_new, E_p_new
    else:
        return state.copy(), E_p





class MCMCLeastSquares(OptimizeLeastSquares):
    '''Markov Chain Monte Carlo sampling of the posterior, assuming all measurement errors are Gaussian (thus the log likelihood becomes a least squares).
        
        Methods:
         * SCAM
         * HMC

        
        HMC:

            Potential energy function is defined as

            .. math

                U(\mathbf{q}) = - \log(\pi(\mathbf{q}) \mathcal{L}(\mathbf{q} | D))

            where \mathbf{q} is the state for inference, \pi is the prior for the state, and L is the liklihood function given the data D


        For SCAM:
         * step: structured numpy array of step sizes for each variable, the steps will be adapted by the algorithm but this is the starting point.
        For HMC:
         * step: structured numpy array with fields: dt (time step) and num (number of leap-frogs in one proposal), and then also "d[variable name]" for the numerical derivate size for that variable.

        #TODO to the gradient analytically since the likelihood is nice function
    '''

    REQUIRED = OptimizeLeastSquares.REQUIRED + [
        'steps',
        'step',
    ]

    OPTIONAL = {
        'method': 'SCAM',
        'method_options': {},
        'prior': None,
        'coord': 'cart',
        'tune': 1000,
        'log_vars': [],
        'proposal': 'normal',
    }

    def __init__(self, data, variables, **kwargs):
        super(MCMCLeastSquares, self).__init__(data, variables, **kwargs)
    
    @mpi_wrap
    def run(self):
        if self.kwargs['start'] is None and self.kwargs['prior'] is None:
            raise ValueError('No start value or prior given.')
        
        start = self.kwargs['start']
        xnow = np.copy(start)
        step = np.copy(self.kwargs['step'])

        steps = self.kwargs['tune'] + self.kwargs['steps']
        chain = np.empty((steps,), dtype=start.dtype)
        
        logpost = self.evalute(xnow)

        if self.kwargs['method'] == 'HMC':

            print('\n{} running {}'.format(type(self).__name__, self.kwargs['method']))
            pbar = tqdm(range(steps), ncols=100)
            for ind in pbar:
                pbar.set_description('Sampling log-posterior = {:<10.3f} '.format(logpost))

                xnew, logpost = hamiltoniuan_monte_carlo(
                    xnow,
                    lambda q: self.evalute(q),
                    lambda q: self._grad(q),
                    step['dt'],
                    step['num'],
                )

                chain[ind] = xnew

        elif self.kwargs['method'] == 'SCAM':
            
            accept = np.zeros((len(self.variables),), dtype=start.dtype)
            tries = np.zeros((len(self.variables),), dtype=start.dtype)
            
            proposal_cov = np.eye(len(self.variables), dtype=np.float64)
            proposal_mu = np.zeros((len(self.variables,)), dtype=np.float64)

            print('\n{} running {}'.format(type(self).__name__, self.kwargs['method']))
            pbar = tqdm(range(steps), ncols=100)
            for ind in pbar:
                pbar.set_description('Sampling log-posterior = {:<10.3f} '.format(logpost))

                xtry = np.copy(xnow)

                pi = int(np.floor(np.random.rand(1)*len(self.variables)))
                var = self.variables[pi]

                proposal = np.random.multivariate_normal(proposal_mu, proposal_cov)
                
                vstep = proposal[pi]*step[0][var]

                if var in self.kwargs['log_vars']:
                    xtry[var] = 10.0**(np.log10(xtry[var]) + vstep)
                else:
                    xtry[var] += vstep
                
                logpost_try = self.evalute(xtry)
                alpha = np.log(np.random.rand(1))
                
                if logpost_try > logpost:
                    _accept = True
                elif (logpost_try - alpha) > logpost:
                    _accept = True
                else:
                    _accept = False
                
                tries[var] += 1.0
                
                if _accept:
                    logpost = logpost_try
                    xnow = xtry
                    accept[var] += 1.0

                if ind % 100 == 0 and ind > 0:
                    for name in self.variables:
                        ratio = accept[0][name]/tries[0][name]

                        if ratio > 0.5:
                            step[0][name] *= 2.0
                        elif ratio < 0.3:
                            step[0][name] /= 2.0
                        
                        accept[0][name] = 0.0
                        tries[0][name] = 0.0
                

                if ind % (steps//100) == 0 and ind > 0:
                    if self.kwargs['proposal'] == 'adaptive':
                        _data = np.empty((len(self.variables), ind), dtype=np.float64)
                        for dim, var in enumerate(self.variables):
                            _data[dim,:] = chain[:ind][var]
                        _proposal_cov = np.corrcoef(_data)
                        
                        if not np.any(np.isnan(_proposal_cov)):
                            proposal_cov = _proposal_cov


                chain[ind] = xnow
            
        chain = chain[self.kwargs['tune']:]

        self.results.trace = chain.copy()
        self._fill_results()

        return self.results


    def _grad(self, state):

        step = self.kwargs['step']

        grad = np.empty((1,), dtype=state.dtype)
        for var in self.variables:
            state_m = state.copy()
            state_p = state.copy()
            
            state_m[0][var] -= step[0]['d' + var]
            state_p[0][var] += step[0]['d' + var]
            
            post_m = self.evalute(state_m)
            post_p = self.evalute(state_p)

            grad[var] = 0.5*(post_p - post_m)/step[0]['d' + var]

        return grad


    def _fill_results(self):
        post_map = np.empty((1,), dtype=self.results.trace.dtype)
        for var in self.variables:
            post_map[var] = np.mean(self.results.trace[var])
        
        self.results.MAP = post_map
        self.loglikelihood(post_map)
        self.results.residuals = copy.deepcopy(self._tmp_residulas)
        self.results.date = self.data['date0']
    




class SMCLeastSquares(Posterior):
    '''Sequential Monte Carlo sampling of the posterior, assuming all measurement errors are Gaussian (thus the log likelihood becomes a least squares).
        
        Methods:
         * PF (Particle filtering)
         * Non-markovian series estimation

    '''
    REQUIRED_DATA = [
        'sources',
        'Model',
        'date0',
        'params',
    ]

    REQUIRED = [
        'propagator',
        'particles',
        'prior',
    ]

    OPTIONAL = {
        'method': 'PF',
        'coord': 'cart',
        'minimum-dt': 0.1, #seconds
        'resample': 'weights',
        'resample-noise': None,
    }

    def __init__(self, data, variables, **kwargs):
        super(SMCLeastSquares, self).__init__(data, variables, **kwargs)
        
        self._state_dtype = [(name, 'float64') for name in self.variables]

        for key in self.REQUIRED:
            if key not in kwargs:
                raise ValueError('Argument "{}" is mandatory'.format(key))

        for key in self.REQUIRED_DATA:
            if key not in data:
                raise ValueError('Data field "{}" is mandatory'.format(key))

        self._models = []

        for source in data['sources']:
            if not isinstance(source, sources.ObservationSource):
                raise ValueError('Non-observation data detected, "{}" not supported'.format(type(source)))

            req_args = copy.copy(self.data['Model'].REQUIRED_DATA)
            for arg in source.avalible_data():
                if arg in req_args:
                    req_args.remove(arg)

            model_data = {
                'propagator': self.kwargs['propagator'],
                'coord': self.kwargs['coord'],
            }
            for arg in req_args:
                if arg not in self.data:
                    raise TypeError('Model REQUIRED data "{}" not found'.format(arg))
                model_data[arg] = self.data[arg]

            model = source.generate_model(
                Model=self.data['Model'],
                **model_data
            )
            self._models.append(model)


    def sample_prior(self, num):

        samples = np.empty((num,), dtype = self._state_dtype)
        for prior in self.kwargs['prior']:
            dist = getattr(scipy.stats, prior['distribution'])
            
            dat = dist.rvs(size=(num,), **prior['params'])
            for name in prior['variables']:
                samples[name] = dat

        return samples


    def resample(self, num):
        pass


    def loglikelihood(self, state, model_index, observation_index):
        '''The loglikelihood function
        '''

        data = self.data['sources'][model_index]
        sim_data = self._models[model_index].evaluate(state)


        logsum = 0.0
        for name, nptype in self._models[model_index].dtype:
            _residuals = tracklets[ind].data[name] - sim_data[name]
            
            logsum -= np.sum(_residuals**2.0/(tracklets[ind].data[name + '_sd']**2.0))
        return 0.5*logsum


    @mpi_wrap_sets
    def run(self):
        _pri_check = []
        for prior in self.kwargs['prior']:
            _pri_check += prior['variables']
        for name in self.variables:
            if name not in _pri_check:
                raise ValueError('Not enough prior information given, "" missing.'.format(name))

        particles = self.kwargs['particles']
        

        chain = self.sample_prior(particles)

        #order data

        dates = []
        for mi, model in enumerate(self._models):
            for di, date in enumerate(model.data['date']):
                dates.append(
                    (date, di, model)
                )
        dates.sort(key=lambda x: x[0]) #sorted according to data arrival

        for obs in range(len(dates)):
            date, di, model = dates[obs]
            while (date - next_date)/np.timedelta64(1.0, 's') < self.kwargs['minimum-dt']:
                pass




def hprint(text):
    print('='*10 + ' ' + text + ' ' + '='*10)


def orbit_determination_fsystem(**kwargs):
    '''Convenience function for running OD.
    
    Assumes propagator works in the same coordinate system.

    **Keyword arguments:**
     
     * state_conversion: function pointer to a function that will convert a state in the frame used by the OEM format into the frame used by the propagator. Will be passed 'mjd0' as keyword argument along with 'state'. Defaults to ITRF to TEME conversion without polar motion.
     * true_object
     * propagator
     * variables
     * output_folder
     * prior_ext
     * tracklet_ext
     * params
     * tracklet_truncate
     * n_tracklets
     * step
     * priors
     * steps
     * tune
    '''
    state_conversion = kwargs.get('state_conversion', None)
    true_space_o = kwargs.get('true_object', None)

    propagator = kwargs.get('propagator', default_propagator_inst)

    variables = kwargs.get('variables', ['x', 'y', 'z', 'vx', 'vy', 'vz'])

    if 'path' not in kwargs:
        raise ValueError('A source path must be given.')
    else:
        _path = kwargs['path']

    fout = kwargs.get('output_folder', None)

    #prop_class = kwargs.get('propagator', default_propagator)
    #prop_args = kwargs.get('propagator_options', {})
    #prop = prop_class(**prop_args)

    prior_paths = sources.Path.recursive_folder(_path, kwargs.get('prior_ext', ['oem']))
    tracklet_paths = sources.Path.recursive_folder(_path, kwargs.get('tracklet_ext', ['tdm']))

    sources_pr = sources.SourceCollection(paths = prior_paths)
    sources_tr = sources.SourceCollection(paths = tracklet_paths)

    ids = list(set([src.index for src in sources_tr]))

    if 'object_id' in kwargs:
        object_id = kwargs['object_id']
    else:
        for ind, _id in enumerate(ids):
            print('[{:<3}]: {}'.format(ind, _id))
        _raw_obj = raw_input('Pick an object: ')
        object_id = ids[int(_raw_obj)]

    sources_pr.filter(object_id)
    sources_tr.filter(object_id)

    if len(sources_pr) > 1:
        hprint('Multiple priors found')
        for ind, path in enumerate(sources_pr):
            print('[{:<3}]: {}'.format(ind, str(path)))
        _raw_obj = raw_input('Pick an prior: ')
        prior_src = sources_pr[int(_raw_obj)]
    else:
        prior_src = sources_pr[0]

    hprint('Priors found')
    for path in sources_pr:
        print(path)
    hprint('Tracklets found')
    for path in sources_tr:
        print(path)

    date0_ind = np.argmin(prior_src.data['date'])
    date0 = prior_src.data[date0_ind]['date']
    state0 = prior_src.data[date0_ind]

    _def_params = {
        'C_D': 2.3,
        'm': 1.0,
        'A': 0.1,
        'C_R': 1.0,
    }

    params = kwargs.get('params', None)
    print('Warning: No parameters given [using default values], may be problematic for orbit determination')
    if params is None:
        params = {}
        for arg in _def_params.keys():
            if arg not in variables:
                params[arg] = _def_params[arg]
    else:
        for arg in _def_params.keys():
            if arg not in variables and arg not in params:
                params[arg] = _def_params[arg]

    hprint('Parameters:')
    for key, item in params.items():
        print('{:<5}: {}'.format(key, item))

    sources_tr.sort(key=lambda x: x.meta['fname'])

    tr_slice = kwargs.get('tracklet_truncate', None)
    pr_slice = kwargs.get('prior_truncate', None)
    n_tracklets = kwargs.get('n_tracklets', None)

    if n_tracklets is not None:
        sources_tr = sources_tr[:n_tracklets]

    if tr_slice is not None:
        for tracklet in sources_tr:
            tracklet.data = tracklet.data[tr_slice]
    
    if pr_slice is not None:
        prior_src.data = prior_src.data[pr_slice]


    if comm.rank == 0:
        if fout is not None:
            if fout[-1] == os.sep:
                fout = fout[:-1]
            with h5py.File(fout + '/{}_obs_data.h5'.format(object_id), 'w') as hf:
                for ind, tracklet in enumerate(sources_tr):
                    hf['{}/mjd'.format(ind)] = dpt.npdt2mjd(tracklet.data['date'])



    variables_orb = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    dt_add = [(name, 'float64') for name in variables if name not in variables_orb]
    new_dt = [(name, 'float64') for name in state0.dtype.names if name in variables_orb and name in variables]
    new_dt += dt_add
    _state0 = np.zeros((1,), dtype=new_dt)

    for var in variables_orb:
        _state0[0][var] = state0[var]

    for name, typ in dt_add:
        if name not in params:
            raise ValueError('No initial value given for parameter "{}" to be estimated.'.format(name))
        else:
            _state0[0][name] = params[name]
            del params[name]
    state0 = _state0

    if state_conversion is None:
        _state0 = _named_to_enumerated(state0, variables_orb)
        _state0 = tle.ITRF_to_TEME(_state0, dpt.mjd_to_jd(dpt.npdt2mjd(date0)), 0.0, 0.0)
        for ind, var in enumerate(variables_orb):
            state0[var] = _state0[ind]
    else:
        state0 = state_conversion(state = state0, mjd=dpt.npdt2mjd(date0))

    input_data_state = {
        'sources': sources_pr,
        'Model': EstimatedState,
        'date0': date0,
        'params': params,
    }

    hprint('Prior estimation start state:')
    for var in variables:
        print('{:<4}: {:<10.3f}'.format(var, state0[var][0]))

    find_prior = OptimizeLeastSquares(
        data = input_data_state,
        variables = variables,
        start = state0,
        prior = None,
        propagator = propagator,
        method = 'Nelder-Mead',
        options = dict(
            maxiter = 10000,
            disp = True,
        ),
    )
    prior_params = find_prior.run()

    hprint('Prior estimation state shift:')
    for var in variables:
        print('{:<4}: {:<14.3f} - {:<14.3f} = {:<10.3f}'.format(var, state0[var][0], prior_params.MAP[var][0], state0[var][0] - prior_params.MAP[var][0]))

    if true_space_o is not None:
        hprint('Prior estimation error (true - estimation):')
        _t_rel = (dpt.npdt2mjd(date0) - true_space_o.mjd0)*3600.0*24.0
        true_prior_state = true_space_o.get_state(_t_rel)

        if state_conversion is None:
            true_prior_state[:,0] = tle.ITRF_to_TEME(true_prior_state[:,0], dpt.mjd_to_jd(dpt.npdt2mjd(date0)), 0.0, 0.0)
        else:
            _state0 = _enumerated_to_named(true_prior_state, variables_orb)
            _state0 = state_conversion(state = _state0, mjd=dpt.npdt2mjd(date0))
            for ind, var in enumerate(variables_orb):
                true_prior_state[ind] = _state0[var]

        for ind, var in enumerate(variables_orb):
            print('{:<4}: {:<14.3f} - {:<14.3f} = {:<10.3f}'.format(var, true_prior_state[ind][0], prior_params.MAP[0][var], true_prior_state[ind][0] - prior_params.MAP[0][var]))

        hprint('Prior start error (true - estimation):')
        for ind, var in enumerate(variables_orb):
            print('{:<4}: {:<14.3f} - {:<14.3f} = {:<10.3f}'.format(var, true_prior_state[ind][0], state0[0][var], true_prior_state[ind][0] - state0[0][var]))

    prior_residuals = find_prior.residuals(prior_params.MAP)

    _res = np.empty((6, len(prior_residuals[0]['residuals'])), dtype=np.float64)
    for ind, var in enumerate(variables_orb):
        _res[ind, :] = prior_residuals[0]['residuals'][var]
    prior_cov = np.cov(_res)

    hprint('Estimated prior covariance')
    print(prior_cov)

    input_data_tracklets = {
        'sources': sources_tr,
        'Model': RadarPair,
        'date0': date0,
        'params': params,
    }

    step = kwargs.get('step', None)
    if step is None:
        step = np.copy(state0)
        for var in ['x', 'y', 'z']:
            if var in variables:
                step[var] = 1.0
        for var in ['vx', 'vy', 'vz']:
            if var in variables:
                step[var] = 0.1
        if 'A' in variables:
            step['A'] = 0.1
        if 'C_D' in variables:
            step['C_D'] = 0.01
        if 'm' in variables:
            step['m'] = 0.1


    prior_dists = kwargs.get('priors', None)
    if prior_dists is None:
        prior_dists = [ #these can be any combination of variables and any scipy continues variable
            {
                'variables': copy.copy(variables_orb),
                'distribution': 'multivariate_normal',
                'params': {
                    'mean': _named_to_enumerated(prior_params.MAP, variables_orb),
                    'cov': prior_cov,
                },
            },
        ]


    find_map = OptimizeLeastSquares(
        data = input_data_tracklets,
        variables = variables,
        start = prior_params.MAP,
        prior = prior_dists,
        propagator = propagator,
        method = 'Nelder-Mead',
        options = dict(
            maxiter = 10000,
            disp = True,
        ),
    )
    
    estimated_map = find_map.run()
    
    hprint('MAP estimation state shift:')
    for var in variables:
        print('{:<4}: {:<14.3f} - {:<14.3f} = {:<10.3f}'.format(var, prior_params.MAP[var][0], estimated_map.MAP[var][0], prior_params.MAP[var][0] - estimated_map.MAP[var][0]))

    if true_space_o is not None:
        hprint('MAP estimation error (true - estimation):')
        for ind, var in enumerate(variables_orb):
            print('{:<4}: {:<14.3f} - {:<14.3f} = {:<10.3f}'.format(var, true_prior_state[ind][0], estimated_map.MAP[0][var], true_prior_state[ind][0] - estimated_map.MAP[0][var]))


    MCMCtrace = MCMCLeastSquares(
        data = input_data_tracklets,
        variables = variables,
        start = estimated_map.MAP,
        prior = prior_dists,
        propagator = propagator,
        method = 'SCAM',
        steps = kwargs.get('steps',100000),
        step = step,
        tune = kwargs.get('tune',3000),
    )
    results = MCMCtrace.run()

    hprint('MCMC mean estimation state shift:')
    for var in variables:
        print('{:<4}: {:<14.3f} - {:<14.3f} = {:<10.3f}'.format(var, estimated_map.MAP[var][0], results.MAP[var][0], estimated_map.MAP[var][0] - results.MAP[var][0]))

    if true_space_o is not None:
        hprint('MAP estimation error (true - estimation):')
        for ind, var in enumerate(variables_orb):
            print('{:<4}: {:<14.3f} - {:<14.3f} = {:<10.3f}'.format(var, true_prior_state[ind][0], results.MAP[0][var], true_prior_state[ind][0] - results.MAP[0][var]))


    if comm.rank == 0:
        if fout is not None:
            if fout[-1] == os.sep:
                fout = fout[:-1]
            results.save(fout + '/{}_orbit_determination.h5'.format(object_id))
    
    dret = {
        'start': state0,
        'mcmc': MCMCtrace,
        'prior': find_prior,
        'MAP-estimation': find_map,
        'results': results,
    }

    return dret


def print_MC_cov(variables, MC_cor, MC_cov):

    print('')
    hprint('MCMC mean estimator correlation matrix')
    print('')

    header = ''.join(['{:<4} |'.format('')] + [' {:<14} |'.format(var) for var in variables])
    print(header)
    for row in MC_cor:
        pr = ''.join(['{:<4} |'.format(row['variable'])] + [' {:<14.6f} |'.format(row[var]) for var in variables])
        print(pr)

    print('')
    hprint('MCMC mean estimator covariance matrix')
    print('')

    header = ''.join(['{:<4} |'.format('')] + [' {:<14} |'.format(var) for var in variables])
    print(header)
    for row in MC_cov:
        pr = ''.join(['{:<4} |'.format(row['variable'])] + [' {:<14.6f} |'.format(row[var]) for var in variables])
        print(pr)



def plot_MC_cov(results, **kwargs):

    _pos = ['x', 'y', 'z']
    _vel = ['vx', 'vy', 'vz']

    MC_cov = results.batch_covariance(kwargs.get('batch_size', len(results.trace)//20))

    MC_cor = MC_cov.copy()
    for xind in range(len(MC_cor)):
        xvar = MC_cor[xind]['variable']
        for yind in range(len(MC_cor)):
            yvar = MC_cor[yind]['variable']
            MC_cor[xind][yvar] /= np.sqrt(MC_cov[xind][xvar]*MC_cov[yind][yvar])

    variance = np.empty((len(results.variables),), dtype=np.float64)
    for xind in range(len(MC_cor)):
        xvar = MC_cor[xind]['variable']
        if xvar in _pos:
            _var = MC_cov[xind][xvar]*1e-3
        else:
            _var = MC_cov[xind][xvar]
        variance[results.variables.index(xvar)] = _var

    print_MC_cov(results.variables, MC_cor, MC_cov)

    fig, ax = plt.subplots(figsize=(15,15))

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap([1,2,3,5,6,7,10])

    labels = kwargs.get('labels', results.variables)
    print(variance)
    _patches, texts = ax.pie(
        variance/np.sum(variance),
        radius=1, 
        colors=outer_colors,
        wedgeprops=dict(
            width=0.3, 
            edgecolor='w',
        ),
        labels = results.variables,
    )

    fig.canvas.draw()

    for ind, xvar in enumerate(labels):
        for yind, yvar in enumerate(labels):
            if xvar != yvar:
                patch1 = [x for x in _patches if str(x._label) == xvar][0]
                patch2 = [x for x in _patches if str(x._label) == yvar][0]

                ang = (patch1.theta2 - patch1.theta1)/2. + patch1.theta1
                y1 = np.sin(np.deg2rad(ang))*0.7
                x1 = np.cos(np.deg2rad(ang))*0.7
                ang = (patch2.theta2 - patch2.theta1)/2. + patch2.theta1
                y2 = np.sin(np.deg2rad(ang))*0.7
                x2 = np.cos(np.deg2rad(ang))*0.7

                verts = [
                    (x1, y1),
                    (0., 0.), 
                    (x2, y2),
                ]

                _with = np.min([MC_cor[yind][results.variables[ind]], MC_cor[ind][results.variables[yind]]])

                pp1 = mpatches.PathPatch(
                    mpath.Path(verts,
                         [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3]),
                    fc="none", transform=ax.transData, linewidth=_with*10,
                    color=outer_colors[ind], alpha = 0.5,
                )
                ax.add_patch(pp1)


    ax.set(aspect="equal", title='Monte-Carlo correlation matrix')
    
    return fig, ax



def plot_scatter_trace(results, **kwargs):

    thin = kwargs.get('thin', None)

    trace2 = results.trace.copy()
    for var in ['x','y','z','vx','vy','vz']:
        if var in results.variables:
            trace2[var] *= 1e-3
    if thin is not None:
        trace2 = trace2[thin]
    df = pandas.DataFrame.from_records(trace2)
    
    cols = kwargs.get('columns', {
        'x':'x [km]',
        'y':'y [km]',
        'z':'z [km]',
        'vx':'$v_x$ [km/s]',
        'vy':'$v_y$ [km/s]',
        'vz':'$v_z$ [km/s]',
        'A':'A [m$^2$]',
    })
    df = df.rename(columns=cols)

    axes = scatter_matrix(df, alpha=kwargs.get('alpha', 0.01), figsize=(15,15))

    if 'fout' in kwargs:
        plt.savefig(kwargs['fout'] + '.png', bbox_inches='tight')

    return axes


def print_covariance(results, **kwargs):

    post_cov = results.covariance()

    print('')
    hprint('Posterior covariance matrix')
    print('')

    header = ''.join(['{:<4} |'.format('')] + [' {:<14} |'.format(var) for var in results.variables])
    print(header)
    for row in post_cov:
        pr = ''.join(['{:<4} |'.format(row['variable'])] + [' {:<14.6f} |'.format(row[var]) for var in results.variables])
        print(pr)



def plot_autocorrelation(results, **kwargs):

    min_k = kwargs.get('min_k', 0)
    max_k = kwargs.get('max_k', len(results.trace)//100)

    MC_gamma = results.autocovariance(min_k = min_k, max_k = max_k)
    Kv = np.arange(min_k, max_k)
    
    fig1, axes1 = plt.subplots(3, 2,figsize=(15,15), sharey=True, sharex=True)
    fig1.suptitle('Markov Chain autocorrelation functions')
    ind = 0
    for xp in range(3):
        for yp in range(2):
            var = results.variables[ind]
            ax = axes1[xp,yp]
            ax.plot(Kv, MC_gamma[var]/MC_gamma[var][0])
            ax.set(
                xlabel='$k$',
                ylabel='$\hat{\gamma}_k/\hat{\gamma}_0$',
                title='Autocorrelation for "{}"'.format(var),
            )
            ind += 1

    fig2, axes2 = plt.subplots(len(results.variables)-6, 1, figsize=(15,15))
    fig2.suptitle('Markov Chain autocorrelation functions')
    for ind, var in enumerate(results.variables[6:]):
        if len(results.variables)-6 > 1:
            ax = axes2[ind]
        else:
            ax = axes2
        ax.plot(Kv, MC_gamma[var]/MC_gamma[var][0])
        ax.set(
            xlabel='$k$',
            ylabel='$\hat{\gamma}_k/\hat{\gamma}_0$',
            title='Autocorrelation for "{}"'.format(var),
        )

    if 'fout' in kwargs:
        fig1.savefig(kwargs['fout'] + '_orbs.png', bbox_inches='tight')
        fig2.savefig(kwargs['fout'] + '_vars.png', bbox_inches='tight')


    return (fig1, axes1), (fig2, axes2)


def plot_trace(results, **kwargs):

    axis_var = kwargs.get('labels', None)
    if axis_var is None:
        axis_var = []
        for var in results.variables:
            if var in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
                axis_var += ['${}$ [km]'.format(var)]
            else:
                axis_var += [var]

    plots = []
    for ind, var in enumerate(results.variables):
        if var in ['x', 'y', 'z', 'vx', 'vy', 'vz']:
            coef = 1e-3
        else:
            coef = 1.0
    
        if ind == 0 or ind == 6:
            if ind == 6 and 'fout' in kwargs:
                fig.savefig(kwargs['fout'] + '_orb.png', bbox_inches='tight')
            fig = plt.figure(figsize=(15,15))
            fig.suptitle(kwargs.get('title','MCMC trace plot'))
            plots.append({
                'fig': plots,
                'axes': []    
            })
            
        if ind <= 5:
            ax = fig.add_subplot(231+ind)
        if ind > 5:
            ax = fig.add_subplot(100*(len(results.variables) - 6) + ind - 5 + 10)
        plots[-1]['axes'].append(ax)
        ax.plot(results.trace[var]*coef)
        ax.set(
            xlabel='Iteration',
            ylabel='{}'.format(axis_var[ind]),
        )
    if 'fout' in kwargs:
        fig.savefig(kwargs['fout'] + '_vars.png', bbox_inches='tight')


    return plots


def plot_residuals(posterior, states, labels, styles, **kwargs):

    residuals = []
    for state in states:
        residuals.append(
            posterior.residuals(state)
        )
    
    plot_n = len(residuals[-1])

    num = len(residuals)

    if plot_n > 3:
        _pltn = 3
    else:
        _pltn = plot_n

    _ind = 0
    for ind in range(plot_n):
        if _ind == _pltn or _ind == 0:
            _ind = 0
            fig = plt.figure(figsize=(15,15))
            fig.suptitle(kwargs.get('title', 'Orbit determination residuals'))

        ax = fig.add_subplot(100*_pltn + 21 + _ind*2)
        for sti in range(num):
            lns = ax.semilogy(
                (residuals[sti][ind]['date'] - residuals[sti][ind]['date'][0])/np.timedelta64(1,'h'),
                np.abs(residuals[sti][ind]['residuals']['r']),
                styles[sti], label=labels[sti], alpha = kwargs.get('alpha',0.5),
            )
        ax.set(
            xlabel='Time [h]',
            ylabel='Range residuals [m]',
            title='Model {}'.format(ind),
        )
        ax.legend()

        ax = fig.add_subplot(100*_pltn + 21+_ind*2+1)
        for sti in range(num):
            lns = ax.semilogy(
                (residuals[sti][ind]['date'] - residuals[sti][ind]['date'][0])/np.timedelta64(1,'h'),
                np.abs(residuals[sti][ind]['residuals']['v']),
                styles[sti], label=labels[sti], alpha = kwargs.get('alpha',0.5),
            )
        ax.set(
            xlabel='Time [h]',
            ylabel='Velocity residuals [m/s]',
            title='Model {}'.format(ind),
        )
        _ind += 1



def plot_orbits(posterior, **kwargs):

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    plothelp.draw_earth_grid(ax)
    ax.set_title(kwargs.get('title', 'Orbit determination: orbital shift'))

    start = kwargs.get('start', None)
    prior = posterior.kwargs['start']
    end = posterior.results.MAP

    states = [start, prior, end]

    _label = [
        'Start state: Simulated measurements',
        'Start state',
        'Prior: Simulated measurements',
        'Prior',
        'Maximum a Posteriori: Simulated measurements',
        'Maximum a Posteriori',
    ]
    _col = [
        'k',
        'b',
        'r',
    ]
    for model in posterior._models:
        for ind, state in enumerate(states):
            states_obs = model.get_states(state)
            _t = model.data['t']
            model.data['t'] = np.linspace(0, np.max(_t), num=kwargs.get('num',1000))
            states = model.get_states(state)
            model.data['t'] = _t
            ax.plot(states_obs[0,:], states_obs[1,:], states_obs[2,:],"."+_col[ind],
                label=_label[ind*2], alpha=kwargs.get('alpha',0.25),
            )
            ax.plot(states[0,:], states[1,:], states[2,:],"-"+_col[ind],
                label=_label[ind*2+1], alpha=kwargs.get('alpha',0.25),
            )
    ax.legend()

    return fig, ax


def plot_orbit_uncertainty(results, true_object, time = 24.0, **kwargs):

    params = {}
    variables = results.variables

    symmetric = kwargs.get('symmetric', False)

    orb_var = ['x', 'y', 'z', 'vx', 'vy', 'vz']

    named_cov = results.covariance()
    cov = np.empty((6,6), dtype=np.float64)

    update_kw = {}
    for var in orb_var:
        if var in variables:
            update_kw[var] = None

    for key, item in true_object.kwargs.items():
        if key not in variables:
            params[key] = item

    so_check = ['C_D', 'A', 'm', 'C_R', 'd']
    for key in so_check:
        if key not in variables:
            params[key] = getattr(true_object, key)

    params.update(kwargs.get('params', {}))

    if symmetric:
        time0 = -time*3600.0
    else:
        time0 = 0.0

    t = np.linspace(time0, time*3600.0, num=kwargs.get('num_t',1000))
    num = kwargs.get('num', 200)

    trace_object = true_object.copy()
    trace_object.mjd0 = dpt.npdt2mjd(results.date)

    t0 = (trace_object.mjd0 - true_object.mjd0)*3600.0*24.0

    obs_dates = kwargs.get('obs_dates', None)
    if obs_dates is not None:
        _num_obs = 0
        for obsi, obs in enumerate(obs_dates):
            obs_dates[obsi] = (dpt.npdt2mjd(obs) - trace_object.mjd0)*24.0
            _num_obs += len(obs)
        _overlap = []
        for xi in range(0, len(obs_dates)):
            for yi in range(xi+1, len(obs_dates)):
                if (np.min(obs_dates[xi]) <= np.max(obs_dates[yi]) and np.min(obs_dates[yi]) <= np.max(obs_dates[xi])):
                    _overlap.append((xi, yi))
        _n_chains = 0
        _groups = [[0]]
        for xi in range(0, len(obs_dates)):
            all_inds = []
            for g in _groups:
                all_inds += g

            if xi in all_inds:
                continue
            else:
                _added = False
                for xj, yj in _overlap:
                    if xj == xi:
                        for g in _groups:
                            if yj in g:
                                g.append(xi)
                                _added = True
                    if yj == xi:
                        for g in _groups:
                            if xj in g:
                                g.append(xi)
                                _added = True
                if not _added:
                    _groups.append([xi])
        _n_chains = len(_groups)
        _text_date = []
        for g in _groups:
            _text_date.append([0, -1e9])
            for gi in g:
                _text_date[-1][0] += len(obs_dates)
                _text_date[-1][1] = np.max([_text_date[-1][1], np.max(obs_dates[gi])])



    true_state = true_object.get_state(t + t0)

    index = np.random.randint(0, len(results.trace), size=(num, ))

    non_orb_variables = [key for key in variables if key not in orb_var]

    for key, item in params.items():
        if key in so_check:
            setattr(trace_object, key, item)
        else:
            trace_object.kwargs[key] = item
    
    r_err = np.empty((num,len(t)), dtype=np.float64)
    v_err = np.empty((num,len(t)), dtype=np.float64)

    for ind in tqdm(range(num)):
        for key in update_kw.keys():
            update_kw[key] = results.trace[index[ind]][key]*1e-3

        trace_object.update(**update_kw)
        for key in non_orb_variables:
            if key in so_check:
                setattr(trace_object, key, results.trace[index[ind]][key])
            else:
                trace_object.kwargs[key] = results.trace[index[ind]][key]
        trace_state = trace_object.get_state(t)

        r_err[ind, :] = np.linalg.norm(true_state[:3,:] - trace_state[:3,:], axis=0)
        v_err[ind, :] = np.linalg.norm(true_state[3:,:] - trace_state[3:,:], axis=0)

    th = t/3600.0

    _mean_r = np.mean(r_err, axis=0)*1e-3
    _std_r = np.std(r_err, axis=0)*1e-3

    fontsize = kwargs.get('fontsize', 18)

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(211)
    if obs_dates is not None:
        ax1.set_title(kwargs.get('title', 'Orbit determination uncertainty: {} (r, v) observations over {} chains'.format(_num_obs, _n_chains)), fontsize = fontsize + 4)
    else:
        ax1.set_title(kwargs.get('title', 'Orbit determination uncertainty', fontsize = fontsize + 4))

    ax1.fill_between(th, _mean_r - _std_r, _mean_r + _std_r, 
        facecolor='b', 
        alpha=kwargs.get('alpha',0.20),
        label='1 sigma error',
    )
    ax1.fill_between(th, _mean_r - 2.0*_std_r, _mean_r + 2.0*_std_r, 
        facecolor='r', 
        alpha=kwargs.get('alpha',0.25),
        label='2 sigma error',
    )

    ax1.plot(th, _mean_r, '-k', alpha=1.0, label='Mean error')
    
    _b, _t = ax1.get_ylim()
    _mid = 0.5*(_b + _t)

    if obs_dates is not None:
        for obs in obs_dates:
            for _d in obs:
                ax1.axvline(_d, alpha=0.5, color='green')
        for _len, _d in _text_date:
            ax1.text(_d + 0.1, _mid, str(_len))

    ax1.legend()
    ax1.set_xlabel('Time [h]', fontsize=fontsize)
    ax1.set_ylabel('Position error [km]', fontsize=fontsize)


    _mean_v = np.mean(v_err, axis=0)
    _std_v = np.std(v_err, axis=0)

    ax1 = fig.add_subplot(212)
    ax1.fill_between(th, _mean_v - _std_v, _mean_v + _std_v, 
        facecolor='b', 
        alpha=kwargs.get('alpha',0.20),
        label='1 sigma error',
    )
    ax1.fill_between(th, _mean_v - 2.0*_std_v, _mean_v + 2.0*_std_v, 
        facecolor='r', 
        alpha=kwargs.get('alpha',0.25),
        label='2 sigma error',
    )

    ax1.plot(th, _mean_v, '-k', alpha=1.0, label='Mean error')

    _b, _t = ax1.get_ylim()
    _mid = 0.5*(_b + _t)

    if obs_dates is not None:
        for obs in obs_dates:
            for _d in obs:
                ax1.axvline(_d, alpha=0.5, color='green')
        for _len, _d in _text_date:
            ax1.text(_d + 0.1, _mid, str(_len))

    ax1.set_xlabel('Time [h]', fontsize=fontsize)
    ax1.set_ylabel('Velocity error [m/s]', fontsize=fontsize)

    ax1.legend()
    


    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(211)
    if obs_dates is not None:
        ax1.set_title(kwargs.get('title', 'Orbit determination uncertainty: {} (r, v) observations over {} chains'.format(_num_obs, _n_chains)), fontsize = fontsize + 4)
    else:
        ax1.set_title(kwargs.get('title', 'Orbit determination uncertainty', fontsize = fontsize + 4))

    ax1.fill_between(th, _mean_r - _std_r, _mean_r + _std_r, 
        facecolor='b', 
        alpha=kwargs.get('alpha',0.20),
        label='1 sigma error',
    )
    ax1.fill_between(th, _mean_r - 2.0*_std_r, _mean_r + 2.0*_std_r, 
        facecolor='r', 
        alpha=kwargs.get('alpha',0.25),
        label='2 sigma error',
    )

    ax1.semilogy(th, _mean_r, '-k', alpha=1.0, label='Mean error')
    
    
    _b, _t = ax1.get_ylim()
    _mid = 0.5*(_b + _t)

    if obs_dates is not None:
        for obs in obs_dates:
            for _d in obs:
                ax1.axvline(_d, alpha=0.5, color='green')
        for _len, _d in _text_date:
            ax1.text(_d + 0.1, _mid, str(_len))

    ax1.legend()
    ax1.set_xlabel('Time [h]', fontsize=fontsize)
    ax1.set_ylabel('Position error [km]', fontsize=fontsize)


    _mean_v = np.mean(v_err, axis=0)
    _std_v = np.std(v_err, axis=0)

    ax1 = fig.add_subplot(212)
    ax1.fill_between(th, _mean_v - _std_v, _mean_v + _std_v, 
        facecolor='b', 
        alpha=kwargs.get('alpha',0.20),
        label='1 sigma error',
    )
    ax1.fill_between(th, _mean_v - 2.0*_std_v, _mean_v + 2.0*_std_v, 
        facecolor='r', 
        alpha=kwargs.get('alpha',0.25),
        label='2 sigma error',
    )

    ax1.semilogy(th, _mean_v, '-k', alpha=1.0, label='Mean error')

    
    _b, _t = ax1.get_ylim()
    _mid = 0.5*(_b + _t)

    if obs_dates is not None:
        for obs in obs_dates:
            for _d in obs:
                ax1.axvline(_d, alpha=0.5, color='green')
        for _len, _d in _text_date:
            ax1.text(_d + 0.1, _mid, str(_len))

    ax1.set_xlabel('Time [h]', fontsize=fontsize)
    ax1.set_ylabel('Velocity error [m/s]', fontsize=fontsize)

    ax1.legend()
    


def plot_orbit_uncertainty_state(estimated_object, covariance, true_object, time = 24.0, **kwargs):

    symmetric = kwargs.get('symmetric', False)

    if symmetric:
        time0 = -time*3600.0
    else:
        time0 = 0.0

    t = np.linspace(time0, time*3600.0, num=kwargs.get('num_t',1000))
    num = kwargs.get('num', 200)

    t0 = (trace_object.mjd0 - true_object.mjd0)*3600.0*24.0

    obs_dates = kwargs.get('obs_dates', None)
    if obs_dates is not None:
        _num_obs = 0
        for obsi, obs in enumerate(obs_dates):
            obs_dates[obsi] = (dpt.npdt2mjd(obs) - trace_object.mjd0)*24.0
            _num_obs += len(obs)
        _overlap = []
        for xi in range(0, len(obs_dates)):
            for yi in range(xi+1, len(obs_dates)):
                if (np.min(obs_dates[xi]) <= np.max(obs_dates[yi]) and np.min(obs_dates[yi]) <= np.max(obs_dates[xi])):
                    _overlap.append((xi, yi))
        _n_chains = 0
        _groups = [[0]]
        for xi in range(0, len(obs_dates)):
            all_inds = []
            for g in _groups:
                all_inds += g

            if xi in all_inds:
                continue
            else:
                _added = False
                for xj, yj in _overlap:
                    if xj == xi:
                        for g in _groups:
                            if yj in g:
                                g.append(xi)
                                _added = True
                    if yj == xi:
                        for g in _groups:
                            if xj in g:
                                g.append(xi)
                                _added = True
                if not _added:
                    _groups.append([xi])
        _n_chains = len(_groups)
        _text_date = []
        for g in _groups:
            _text_date.append([0, -1e9])
            for gi in g:
                _text_date[-1][0] += len(obs_dates)
                _text_date[-1][1] = np.max([_text_date[-1][1], np.max(obs_dates[gi])])



    true_state = true_object.get_state(t + t0)
    
    sobj = estimated_object.copy()

    r_err = np.empty((num,len(t)), dtype=np.float64)
    v_err = np.empty((num,len(t)), dtype=np.float64)

    for ind in tqdm(range(num)):

        xi = scipy.stats.multivariate_normal.rsv(cov=cov)
        sobj.update(
            x = sobj.x + xi[0],
            y = sobj.y + xi[1],
            z = sobj.z + xi[2],
            vx = sobj.vx + xi[3],
            vy = sobj.vy + xi[4],
            vz = sobj.vz + xi[5],
        )
        trace_state = sobj.get_state(t)

        r_err[ind, :] = np.linalg.norm(true_state[:3,:] - trace_state[:3,:], axis=0)
        v_err[ind, :] = np.linalg.norm(true_state[3:,:] - trace_state[3:,:], axis=0)

    th = t/3600.0

    _mean_r = np.mean(r_err, axis=0)*1e-3
    _std_r = np.std(r_err, axis=0)*1e-3

    fontsize = kwargs.get('fontsize', 18)

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(211)
    if obs_dates is not None:
        ax1.set_title(kwargs.get('title', 'Orbit determination uncertainty: {} (r, v) observations over {} chains'.format(_num_obs, _n_chains)), fontsize = fontsize + 4)
    else:
        ax1.set_title(kwargs.get('title', 'Orbit determination uncertainty', fontsize = fontsize + 4))

    ax1.fill_between(th, _mean_r - _std_r, _mean_r + _std_r, 
        facecolor='b', 
        alpha=kwargs.get('alpha',0.20),
        label='1 sigma error',
    )
    ax1.fill_between(th, _mean_r - 2.0*_std_r, _mean_r + 2.0*_std_r, 
        facecolor='r', 
        alpha=kwargs.get('alpha',0.25),
        label='2 sigma error',
    )

    ax1.plot(th, _mean_r, '-k', alpha=1.0, label='Mean error')
    
    _b, _t = ax1.get_ylim()
    _mid = 0.5*(_b + _t)

    if obs_dates is not None:
        for obs in obs_dates:
            for _d in obs:
                ax1.axvline(_d, alpha=0.5, color='green')
        for _len, _d in _text_date:
            ax1.text(_d + 0.1, _mid, str(_len))

    ax1.legend()
    ax1.set_xlabel('Time [h]', fontsize=fontsize)
    ax1.set_ylabel('Position error [km]', fontsize=fontsize)


    _mean_v = np.mean(v_err, axis=0)
    _std_v = np.std(v_err, axis=0)

    ax1 = fig.add_subplot(212)
    ax1.fill_between(th, _mean_v - _std_v, _mean_v + _std_v, 
        facecolor='b', 
        alpha=kwargs.get('alpha',0.20),
        label='1 sigma error',
    )
    ax1.fill_between(th, _mean_v - 2.0*_std_v, _mean_v + 2.0*_std_v, 
        facecolor='r', 
        alpha=kwargs.get('alpha',0.25),
        label='2 sigma error',
    )

    ax1.plot(th, _mean_v, '-k', alpha=1.0, label='Mean error')

    _b, _t = ax1.get_ylim()
    _mid = 0.5*(_b + _t)

    if obs_dates is not None:
        for obs in obs_dates:
            for _d in obs:
                ax1.axvline(_d, alpha=0.5, color='green')
        for _len, _d in _text_date:
            ax1.text(_d + 0.1, _mid, str(_len))

    ax1.set_xlabel('Time [h]', fontsize=fontsize)
    ax1.set_ylabel('Velocity error [m/s]', fontsize=fontsize)

    ax1.legend()
    


    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(211)
    if obs_dates is not None:
        ax1.set_title(kwargs.get('title', 'Orbit determination uncertainty: {} (r, v) observations over {} chains'.format(_num_obs, _n_chains)), fontsize = fontsize + 4)
    else:
        ax1.set_title(kwargs.get('title', 'Orbit determination uncertainty', fontsize = fontsize + 4))

    ax1.fill_between(th, _mean_r - _std_r, _mean_r + _std_r, 
        facecolor='b', 
        alpha=kwargs.get('alpha',0.20),
        label='1 sigma error',
    )
    ax1.fill_between(th, _mean_r - 2.0*_std_r, _mean_r + 2.0*_std_r, 
        facecolor='r', 
        alpha=kwargs.get('alpha',0.25),
        label='2 sigma error',
    )

    ax1.semilogy(th, _mean_r, '-k', alpha=1.0, label='Mean error')
    
    
    _b, _t = ax1.get_ylim()
    _mid = 0.5*(_b + _t)

    if obs_dates is not None:
        for obs in obs_dates:
            for _d in obs:
                ax1.axvline(_d, alpha=0.5, color='green')
        for _len, _d in _text_date:
            ax1.text(_d + 0.1, _mid, str(_len))

    ax1.legend()
    ax1.set_xlabel('Time [h]', fontsize=fontsize)
    ax1.set_ylabel('Position error [km]', fontsize=fontsize)


    _mean_v = np.mean(v_err, axis=0)
    _std_v = np.std(v_err, axis=0)

    ax1 = fig.add_subplot(212)
    ax1.fill_between(th, _mean_v - _std_v, _mean_v + _std_v, 
        facecolor='b', 
        alpha=kwargs.get('alpha',0.20),
        label='1 sigma error',
    )
    ax1.fill_between(th, _mean_v - 2.0*_std_v, _mean_v + 2.0*_std_v, 
        facecolor='r', 
        alpha=kwargs.get('alpha',0.25),
        label='2 sigma error',
    )

    ax1.semilogy(th, _mean_v, '-k', alpha=1.0, label='Mean error')

    
    _b, _t = ax1.get_ylim()
    _mid = 0.5*(_b + _t)

    if obs_dates is not None:
        for obs in obs_dates:
            for _d in obs:
                ax1.axvline(_d, alpha=0.5, color='green')
        for _len, _d in _text_date:
            ax1.text(_d + 0.1, _mid, str(_len))

    ax1.set_xlabel('Time [h]', fontsize=fontsize)
    ax1.set_ylabel('Velocity error [m/s]', fontsize=fontsize)

    ax1.legend()
    
