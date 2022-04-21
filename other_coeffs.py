import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

pearson = lambda x, y : np.abs(stats.pearsonr(x, y)[0]) 
pearson.__name__ = 'pearson' 

spearman = lambda x, y : np.abs(stats.spearmanr(x, y)[0]) 
spearman.__name__ = 'spearman'

kendall = lambda x, y : np.abs(stats.kendalltau(x, y)[0])
kendall.__name__ = 'kendall'

def dcor(x, y,  d_x_function='euclidean', d_y_function='euclidean'):
    """ Compute the distance correlation function
    """
    n = x.shape[0]
    if len(x.shape) < 2:
        x = x.reshape(n, 1)
    if len(y.shape) < 2:
        y = y.reshape(n, 1)

    d_x = cdist(x, x, d_x_function)
    d_y = cdist(y, y, d_y_function)

    # keepdims to allow broadcasting.
    U = d_x - d_x.mean(axis=0, keepdims=True) - d_x.mean(axis=1, keepdims=True) + d_x.mean()
    V = d_y - d_y.mean(axis=0, keepdims=True) - d_y.mean(axis=1, keepdims=True) + d_y.mean()

    # term 1/n**2 is missing in these referencings since it simplifies.
    dcov2_xy = (U * V).sum()
    dcov2_xx = (U * U).sum()
    dcov2_yy = (V * V).sum()
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

