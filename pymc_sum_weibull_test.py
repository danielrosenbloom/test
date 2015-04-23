"""
Each observation is the sum of two underlying parameters (x = t1 + t2).
There are n observations in all.

t1 ~ Weibull(shape1, scale1)
t2 ~ Weibull(shape2, scale2)

I want to estimate the four population parameters shape1, scale1, shape2, scale2.
In addition, I want to estimate t1[i] for each observation i.
"""

import pymc as pm
import numpy as np
np.random.seed(8675309) # for reproducibility

# Uniform priors for weibull shapes & scales:
SHAPE1_MIN = 0
SHAPE1_MAX = 10
SHAPE2_MIN = 0
SHAPE2_MAX = 10

SCALE1_MIN = 0
SCALE1_MAX = 100
SCALE2_MIN = 0
SCALE2_MAX = 100

ITER = 100000 # Total iterations
THIN = 100 # thinning interval
SAVE = 10000 # save interval

def model(n, shape1, scale1, shape2, scale2, data=None):
    """Creates model for fitting elements of data to sum of two weibulls
    If data is None, then builds model for simulation
    
    Returns list of pymc objects representing the desired model.
    """
    t1 = np.empty(n, dtype=object) # first weibull
    t2 = np.empty(n, dtype=object) # second weibull
    x = np.empty(n, dtype=object) # sum -- this is the observed data
    
    if data is not None:
        # Model for inference from observed sum:
        
        for i in xrange(n):
            x[i] = pm.Index('x_%i'%i, x=data, index=i) # x[i] is Deterministic, given by the observed data
            t1[i] = pm.Weibull('t1_%i'%i, alpha=shape1, beta=scale1)
            # Ensure that initial guess for t1 is not more than the observed sum:
            if t1[i].value >= x[i].value:
                t1[i].value = 0.95 * x[i].value
            #/if
        #next i
        
        # Now define t2 = x - t1:
        for i in xrange(n):
            def subtractfunc(t1=t1, x=x, ii=i):
                return x[ii] - t1[ii]
            #/def subtractfunc
            t2[i] = pm.Lambda('t2_%i'%i, subtractfunc) # t2[i] is Deterministic (fully determined by x[i] and t1[i])
        #next i
        
        # Now define the probability density for t2 (Weibull with shape2, scale2)
        t2dist = np.empty(n, dtype=object)
        for i in xrange(n):
            def weibfunc(t2=t2, shape2=shape2, scale2=scale2, ii=i):
                return pm.weibull_like(t2[ii], alpha=shape2, beta=scale2)
            #/def weibfunc
            t2dist[i] = pm.Potential(logp = weibfunc,
                                       name = 't2dist_%i'%i,
                                       parents = {'shape2':shape2, 'scale2':scale2, 't2':t2},
                                       doc = 'weibull potential for t2',
                                       verbose = 0,
                                       cache_depth = 2)
        #next i
        
        return [shape1, scale1, shape2, scale2, t1, t2, x, t2dist]
    else: # data is None -- so we treat the shapes and scales as known values
        for i in xrange(n):
            t1[i] = pm.Weibull('t1_%i'%i, alpha=shape1, beta=scale1)
            t2[i] = pm.Weibull('t2_%i'%i, alpha=shape2, beta=scale2)
            x[i] = t1[i] + t2[i]
        #next i
        
        return [t1, t2, x]
    #/if
#/def model

def simulate_data(n, shape1, scale1, shape2, scale2):
    """Returns list of simulated observed data
    """
    sim_model = model(n, shape1, scale1, shape2, scale2) # returns [t1, t2, x]
    return [xi.value for xi in sim_model[2]] # the initial value is a random draw
#/def

def fit_model(data, filename = 'pm.pickle'):
    """Uses AM MCMC
    """
    shape1 = pm.Uniform('shape1', SHAPE1_MIN, SHAPE1_MAX)
    shape2 = pm.Uniform('shape2', SHAPE2_MIN, SHAPE2_MAX)
    scale1 = pm.Uniform('scale1', SCALE1_MIN, SCALE1_MAX)
    scale2 = pm.Uniform('scale2', SCALE2_MIN, SCALE2_MAX)
    
    n = len(data)
    
    all_params = model(n, shape1, scale1, shape2, scale2, data=data)
    stoch_params = all_params[:4] + all_params[4].tolist() # Flattened list of all stochastic parameters to be updated with adaptive metropolis -- this is n+4 parameters in all
    
    M = pm.MCMC(all_params, db='pickle', dbname=filename)
    M.use_step_method(pm.AdaptiveMetropolis, stoch_params)
    M.sample(iter=ITER, thin=THIN, save_interval=SAVE, burn_till_tuned=True, tune_throughout=True, tune_interval=int(max(1000,ITER/20)))
    M.db.close()
    
    return M
#/def fit_model

def sim_and_test(n, shape1, scale1, shape2, scale2):
    """Simulates data and then tries to fit the model.
    """
    data = simulate_data(n, shape1, scale1, shape2, scale2)
    basename = '_'.join(str(s) for s in [n,shape1,scale1,shape2,scale2])
    filename_pickle = basename + '.pickle'
    filename_csv = basename + '.csv'
    M = fit_model(data, filename_pickle)
    M.write_csv(filename_csv, variables = ['shape1', 'scale1', 'shape2', 'scale2'])
    
    f = open(filename_csv, 'r')
    lines = [line[:-1] for line in f.readlines()]
    lines[0] += ', TRUE VALUE\n'
    lines[1] += ', ' + str(shape1) + '\n'
    lines[2] += ', ' + str(scale1) + '\n'
    lines[3] += ', ' + str(shape2) + '\n'
    lines[4] += ', ' + str(scale2) + '\n'
    f.close()
    
    f = open(filename_csv, 'w')
    f.writelines(lines)
    f.close()
    
    return M
#/def sim_and_test

#######################################
#### MAIN

M=sim_and_test(60,  # number of observations
               1.0, # shape 1
               30., # scale 1
               6.5, # shape 2
               10.) # scale 2
pm.Matplot.plot(M.trace('shape1'))
pm.Matplot.plot(M.trace('shape2'))
pm.Matplot.plot(M.trace('scale1'))
pm.Matplot.plot(M.trace('scale2'))
