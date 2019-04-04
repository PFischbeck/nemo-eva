import numpy as np
import powerlaw
import helpers.tail_estimation as pl
import networkit
import sys
import random
import math

#def powerlaw_fit(degrees):
#    fit = powerlaw.Fit(degrees, fit_method='Likelihood', verbose=False)
#    return fit.alpha


def powerlaw_fit(degrees):
    degrees = np.array(degrees)
    # Apply noise
    degrees = pl.add_uniform_noise(degrees, p=1)
    degrees[::-1].sort()
    alphas = []
    try:
        result = pl.hill_estimator(degrees)
        alpha = 1 + 1 / result[3]
        alphas.append(alpha)
    except:
        pass
    
    try:
        result = pl.moments_estimator(degrees)
        alpha = 1 + 1 / result[3]
        alphas.append(alpha)
    except:
        pass
        
    try:    
        result = pl.kernel_type_estimator(degrees, hsteps=200)
        alpha = 1 + 1 / result[3]
        alphas.append(alpha)
    except:
        pass    
    
    if alphas:
        alpha = np.median(np.array(alphas))
    else:
        alpha = 2.1
    alpha = max(alpha, 2.1)
    return alpha


# Return a power-law distribution
def powerlaw_generate(n, k, gamma):
    #generator = networkit.generators.PowerlawDegreeSequence(1, max_deg, -gamma)
    
    #generator.setGamma(-gamma)
    #generator.run()
    #generator.setMinimumFromAverageDegree(max(generator.getExpectedAverageDegree(), k))
    #generator.run()
    #print("powerlaw: wanted k={}, but will get {}".format(k, generator.getExpectedAverageDegree()), file=sys.stderr)

    #degrees = generator.getDegreeSequence(n)
    '''
    wanted_m2 = n * k
    current_m2 = sum(degrees)

    if current_m2 > wanted_m2:
        diff = int(current_m2 - wanted_m2)
        modifier = -1
    else:
        diff = int(wanted_m2 - current_m2)
        modifier = +1
    
    while diff:
        pos = random.randrange(n)
        if degrees[pos] + modifier >= 0:
            degrees[pos] += modifier
            diff -= 1
    '''
    min_deg = 1
    max_deg = n-1
    min_pow = min_deg ** (-gamma+1)
    max_pow = max_deg ** (-gamma+1)

    degrees = np.array([0.]*n)
    for i in range(n):
        deg = ((max_pow - min_pow) * random.random() + min_pow)** (1 / (-gamma+1))
        degrees[i] = deg

    factor = k * n / sum(degrees)

    degrees *= factor
    degrees = np.around(degrees)

    #print("powerlaw: min {}, max {}".format(generator.getMinimumDegree(), generator.getMaximumDegree()), file=sys.stderr)
    print("powerlaw: wanted k={}, will get {}".format(k, sum(degrees) / n, file=sys.stderr))

    return degrees