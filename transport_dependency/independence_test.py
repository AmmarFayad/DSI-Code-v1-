import numpy as np
from transport_dependency import tcor, mtcor

def permutation_test(x, y, coef=tcor, level=0.05, m=99, exact_level=True, **args):
    """ 
    A permutatoin test for indipendence using m permutations. The null hypotesis of the test is
    that x and y are samples drawn form an independent distribution. 

    Note that if exact_level=True, this test is exactly of the given level and sometimes
    the test rejects even if the p_value is greater than the level of the test. In order
    to avoid such artifacts we suggest to always choose m such that (m+1)*level is an integer value.

    Parameters
    ----------
    x    : array_like
    y    : array_like
    coef : function
         A dependency coefficient to evaluate the dependency between x and y. By default the
         transport correlation is used
    level: float 
         The level of the test
    m    : int
        Number of resamples 
    exact_level : bool
        If True then the test is exactly of specified level by using a randomized rejection.
        If False the test rejects whenever the p-value is under the specified level

    args : additoinal arguments that can be passed to the function coef
    
    Returns
    -------
    p_val     : float 
              Computed p_value
    test_res  : int
              The value 1 means that that the independence hypotesis can be rejected.
              The value 0 implies that the hypotesis could not be rejected. 
    test_stat : float
              Value of the test statisitcs, given by coef(x, y, **args)
    """

    test_stat = coef(x, y, **args)
    # m random permutations of y (this could be actually executed in parallel)
    perm_y = np.array([np.random.permutation(y) for _ in range(m)])

    # evaluating coef on the permuted data.
    t_ind = np.array([coef(x, perm_y[i], **args) for i in range(m)])
    
    # we now add t to the other independent coefficients 
    t_mixed = np.append(t_ind, test_stat)

    # Index of the 1-alpha-quantile where alpha is the level of the test.
    # The "-1" is necessary since indexing starts from 0 and not 1. 
    quantile_index = np.int(np.ceil((1-level)*(len(t_mixed)))) - 1
    quantile = np.sort(t_mixed)[quantile_index]

    if test_stat > quantile:
        test_result = 1

    elif test_stat < quantile:
        test_result = 0

    elif test_stat == quantile:
        if exact_level:
            # probability of t > quantile (under null hypotesis)
            prq = np.sum(t_mixed > quantile) / len(t_mixed) 

            # probability of t = q
            prob_q = np.sum(t_mixed==quantile)/len(t_mixed) 

            # prob_r = ((1 - level) - prq) / prob_q
            prob_r = (level - prq) / prob_q
            # a probabilistic decision is made in order to derive a test of the desired level 
            test_result = np.random.binomial(1, prob_r) 
        else:
            test_result = 0

    # the p-value is given by the probability of seeing t >= t_ind under the null hypotesis
    p_value = (np.sum(t_mixed>=test_stat)) / (m+1) 
    return {"test_res" : test_result, "p_val" : p_value, "test_stat" : test_stat}

