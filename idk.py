import torch
import numpy as np

def knn(x, y, k=1, last_only=False, discard_nearest=True):
    """Find k_neighbors-nearest neighbor distances from y for each example in a minibatch x.
    :param x: tensor of shape [T, N]
    :param y: tensor of shape [T', N]
    :param k: the (k_neighbors+1):th nearest neighbor
    :param last_only: use only the last knn vs. all of them
    :param discard_nearest:
    :return: knn distances of shape [T, k_neighbors] or [T, 1] if last_only
    """

    dist_x = (x ** 2).sum(-1).unsqueeze(1)  # [T, 1]
    dist_y = (y ** 2).sum(-1).unsqueeze(0)  # [1, T']
    cross = - 2 * torch.mm(x, y.transpose(0, 1))  # [T, T']
    distmat = dist_x + cross + dist_y  # distance matrix between all points x, y
    distmat = torch.clamp(distmat, 1e-8, 1e+8)  # can have negatives otherwise!

    if discard_nearest:  # never use the shortest, since it can be the same point
        knn, _ = torch.topk(distmat, k + 1, largest=False)
        knn = knn[:, 1:]
    else:
        knn, _ = torch.topk(distmat, k, largest=False)

    if last_only:
        knn = knn[:, -1:]  # k_neighbors:th distance only

    return torch.sqrt(knn)


def kl_div(x, y, k=1, eps=1e-8, last_only=False):
    """KL divergence estimator for batches x~p(x), y~p(y).
    :param x: prediction; shape [T, N]
    :param y: target; shape [T', N]
    :param k:
    :return: scalar
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x.astype(np.float32))
        y = torch.tensor(y.astype(np.float32))

    nns_xx = knn(x, x, k=k, last_only=last_only, discard_nearest=True)
    nns_xy = knn(x, y, k=k, last_only=last_only, discard_nearest=False)

    divergence = (torch.log(nns_xy + eps) - torch.log(nns_xx + eps)).mean()

    return divergence


def entropy(x, k=1, eps=1e-8, last_only=False):
    """Entropy estimator for batch x~p(x).
        :param x: prediction; shape [T, N]
        :param k:
        :return: scalar
        """
    if type(x) is np.ndarray:
        x = torch.tensor(x.astype(np.float32))

    nns_xx = knn(x, x, k=k, last_only=last_only, discard_nearest=True)

    ent = torch.log(nns_xx + eps).mean() - torch.log(torch.tensor(eps))

    return ent


import numpy.random as nr
x = nr.normal(0,1,[100,1])
y = nr.normal(0,1,[100,1])
print(entropy(y)+entropy(x)-entropy(np.concatenate((x,y),axis=1)))
# print(x)

S=torch.FloatTensor(1,10)
print(S.t())
print(entropy(S.t()))

data = np.random.randn(5, 1*1000)
X = data[0:2,:]
# X= torch.tensor(X.astype(np.float32))
data = np.random.rand(10, 1*1000)
Y = data[0:2,:]
# Y= torch.tensor(Y.astype(np.float32))
print(entropy(np.concatenate((X,Y),axis=1)),entropy(Y))


