import ot
import numpy as np
from scipy.spatial.distance import pdist, squareform

def tcor(x, y, p=1, r=1, alpha=None, gamma=None, d_x='euclidean', d_y='euclidean',est_ind=False , solver=ot.emd2, **solver_args):
    """
    Compute the transport correlation.

    Parameters
    ----------
    x : array_like
    y : array_like
    p : float, determines the power to be used 
    r : float, determines the r-product metric 
    alpha : float, if alpha is None then alpha* is computed
    gamma : (n_x, n_y) matrix, the joint distribution of two random variables. 
    d_x : distance function (needs to be at least symmetric)
    d_y : distance function (needs to be at least symmetric)
    est_ind : bool, determines weather the independent coupling should be estimated by len(x) data points using a random permuatoins
    solver : function, the optimal transport solver to be used
    solver_args : additional arguments for the optimal transport solver

    Returns
    -------
    transport correlation : float
        
    Examples
    --------
    >>> import numpy
    >>> d = 2
    >>> n = 20 
    >>> x = np.random.multivariate_normal(np.zeros(d), np.eye(d), n)
    >>> y = 3 * x + [2, 7]
    >>> res = tcor(x, y)
    >>> print(res)


    Reference
    --------
    [1] Transport Dependency: Optimal Transport Based Dependency Measures, Thomas Giacomo Nies, Thomas Staudt and Axel Munk, 2021
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if gamma is None:   # If gamma is None we assume x and y to be empirical data with uniform weight.
        n = len(x)
        mu = np.ones(n)/n
        nu = np.ones(n)/n
    else:
        mu, nu = marginals(gamma)

    d_x = distance_matrix(x, d_x)
    d_y = distance_matrix(y, d_y)

    pdiam_x = diameter(d_x, mu, p)
    pdiam_y = diameter(d_y, nu, p)

    if alpha is None: # determine alpha_star
        alpha = pdiam_y/pdiam_x

    c_x = (alpha *d_x)**r  
    c_y = d_y**r 

    h = lambda x : x**(p/r)/ pdiam_y # Here we are already scaling by the diameter of \nu\otimes\nu.

    if est_ind:
        if gamma is not None:
            raise Exception("Estimation of independent couplig is implemented only for empirical measures")
        res = td_permutation_estimator(c_x, c_y, h, solver=solver, **solver_args)**(1/p)
    else:
        res = td(c_x, c_y, h, gamma, solver=solver, **solver_args)**(1/p) 
    return res

def mtcor(x, y, p=1, gamma=None, d_x='euclidean', d_y='euclidean', solver=ot.emd2, **solver_args):
    """
    Compute the marginal transport correlation. It is here assumed that x is categorical.
    Parameters
    ----------
    x : array_like
        needs to be categorical
    y : array_like
    p : float
    gamma : (n_x, n_y) matrix
    d_y : distance function (needs to be at least symmetric)
    solver : optimal transport solver
    solver_args : additional arguments for the optimal transport solver

    Returns
    -------
    transport correlation : float

    Reference
    --------
    [1] Transport Dependency: Optimal Transport Based Dependency Measures, Thomas Giacomo Nies, Thomas Staudt and Axel Munk, 2021
    """
    if gamma is None:   # If gamma is None we assume x and y to be empirical data with uniform weight.
        x_cat = np.unique(x)
        n_x = len(x_cat) 
        n_y = len(y)
        gamma = (x_cat[:, np.newaxis] == x) / n_y 

    mu, nu = marginals(gamma)

    d_y = distance_matrix(y, d_y)
    pdiam_y = diameter(d_y, nu, p)
    c_y = d_y**p / pdiam_y

    res = mtd(c_y, gamma, solver=solver, **solver_args)**(1/p) 
    return res

def td(c_x, c_y, h, gamma=None, solver=ot.emd2, **solver_args):
    """
    Computes the transport dependence given by \tau(\gamma) = OT_c(\gamma, \mu\otimes\nu) 
    for costs on the product space of the form c(x, y, x', y') = h(c_x(x, x') + c_y(y, y')).
    where :

    - gamma is a distribution on a product space X\times Y 
    - c_x is the cost function on X
    - c_y is the cost function on Y
    - h is real valued function with real domain

    Parameters
    ----------
    c_x : array_like
        cost matrix on the first marginal space. The value c_x[i, k] should be given by c_x(x_i, x'_k). 
    c_y : array_like
        cost matrix on the second marginal space. The value c_y[j, l] should be given by c_y(y_j, y'_l). 
    h   : function 
        function must support broadcasting
    gamma : array_like
        probability simplex of the coupling, if none then the empirical distribution 
        \gamma = 1/n \sum_{i=1}^n \delta_{x_i, y_i} is assumed.
    solver : function, determines the solver to be used to actually solve underlying transport problem.
    solver_args : additional arguments for the optimal transport solver
    
    Returns
    -------
    res : float 
    Note that different solvers might generate a different output.
    """
    n_x = c_x.shape[0]
    n_y = c_y.shape[0]
    if gamma is None:   # If gamma is None we assume equal weights.
                        # This implies that n_x should be equal to n_y.
        n = n_x
        gamma = np.ones(n)/n
        independent = np.ones((n**2))/n**2
        c = cost_on_product_space_3d(c_x, c_y, h)
    else:
        independent = independent_coupling(gamma)
        gamma = gamma.reshape((-1))
        c = cost_on_product_space_4d(c_x, c_y, h)
    res = solver(independent, gamma, c, **solver_args)
    return res

def mtd(c_y, gamma, solver=ot.emd2, **solver_args):
    r"""
    Retruns the marginal transport dependency
    Parameters
    ----------
    gamma : (n_x, n_y) matrix
            This matrix needs to have postitive values
    c_y   : (n_y, n_y) matrix
            the cost matrix on the second marginal space
    solver : function, optimal transport solver
    solver_args : additional arguments for the optimal transport solver

    Returns
    -------
    transport correlation : float
        
    Examples
    """
    n_x, n_y = gamma.shape
    mu, nu = marginals(gamma)
    res = np.sum(np.array([solver(mu[i]*nu, gamma[i, :], c_y, **solver_args) for i in range(n_x)]))
    return res

def td_permutation_estimator(c_x, c_y, h, solver=ot.emd2):
    """Approximate the transport transport dependency by estimating \mu\otimes\nu with measure with n support points obtained by a random permutatoin."""
    n_x = c_x.shape[0]
    n_y = c_y.shape[0]
    n = n_x
    gamma = np.ones(n)/n
    perm_y = np.random.permutation(n_y)
    # perm_inv = np.argsort(perm_y)
    c = np.ascontiguousarray(c_x + c_y[:, perm_y])
    mutimesnu_est = np.ones(n)/n
    
    res = solver(mutimesnu_est, gamma, c, **solver_args)
    return res

def cost_on_product_space_3d(c_x, c_y, h):
    """Create a cost matrix on product space.
    If n = c_x.shape[0] then c[k, l] corresponds to the transportation cost h(c_x(x_i, x_l) + c_y(y_j, y_l)) with i and j determined by l = i * n + j.

    Keyword arguments:
    c_x   -- The cost matrix on X 
    c_y   -- The cost matrix on Y 
    h     -- Vectorized function
    """
    n = c_x.shape[0]
    c = h(c_x.reshape((n, 1, n)) + c_y.reshape((1,  n, n))) # broadcasting over last dimension
    c = c.reshape((n**2, n))   
    return c

def cost_on_product_space_4d(c_x, c_y, h):
    """Create a cost matrix on product space.
    If n_x = c_x.shape[0] and n_y = c_y.shape[0] then c[u, v] corresponds to the transportation cost h(c_x(x_i, x_k) + c_y(y_j, y_l))
    with i and j determined by u = i * n_x + j and k, l determined by v = j * n_y + l.

    Keyword arguments:
    c_x   -- The cost matrix on X 
    c_y   -- The cost matrix on Y 
    h     -- Vectorized function
    """
    n_x = c_x.shape[0]
    n_y = c_y.shape[0]
    c = h(c_x.reshape((n_x, 1, n_x, 1)) + c_y.reshape((1,  n_y, 1,  n_y)))
    c = c.reshape((n_x*n_y, n_x*n_y))   
    return c

def marginals(gamma):
    """ Computes histogram of marginals given a histogram on product space."""
    return np.sum(gamma, axis=1), np.sum(gamma, axis=0) 

def independent_coupling(gamma):
    """ Computes histogram of marginals given a histogram on product space."""
    mu, nu = marginals(gamma)
    indep = mu.reshape(-1, 1) * nu.reshape(1, -1)
    return indep.reshape((-1)) 

def diameter(d, mu, p):
    """Computes the p-diameter given a distance matrix.""" 
    return np.einsum("i, ij, j", mu, d**p, mu)

def distance_matrix(points, d_function):
    """ Computes pairwise distance between points."""
    n = len(points)
    if len(points.shape) < 2:
        points = points.reshape(n, 1)
    d = squareform(pdist(points, d_function))
    return d 

def alpha_star(x, y, d_x="euclidean", d_y="euclidean", p=1):
    """ Computes alpha star"""
    d_x = distance_matrix(x, d_x)
    d_y = distance_matrix(y, d_y)
    mu = np.ones(len(x))/len(x)
    nu = np.ones(len(y))/len(y)

    pdiam_x = diameter(d_x, mu, p)
    pdiam_y = diameter(d_y, nu, p)

    alpha = pdiam_y/pdiam_x

    return alpha
