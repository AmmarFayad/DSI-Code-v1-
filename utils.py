import numpy as np
import torch
import math
import continuous
from sklearn.neighbors import NearestNeighbors
###

def entropy_KL1(data,k=1):
    #KL entropy estimator, with k=1 nearest neighbor
    
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data.transpose())
    distances, indices = nbrs.kneighbors(data.transpose())
    dist_nearest = (distances[:,1]).transpose()
    if data.shape[0] == 1:
        vols_nearest = 2*dist_nearest
    elif data.shape[0] == 2:
        vols_nearest = np.pi * (dist_nearest)**2
    Hhat = np.log(k) + .5772156 + (1/data.shape[1]) * np.sum(np.log(data.shape[1]/k*vols_nearest))
    return Hhat

def rand_slices(dim, num_slices=1000):
    slices = torch.randn((num_slices, dim))
    slices = slices / torch.sqrt(torch.sum(slices ** 2, dim=1, keepdim=True))
    return slices

def arccos_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(np.arccos(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps))))

def I_est(x,y):
    # return continuous.get_mi_mvn(x,y)
    return -entropy_KL1(np.vstack((x, y))) + entropy_KL1(x) + entropy_KL1(y)

def DSI (X, Y, num_slices, f1,f2, f1_op,f2_op, omega_X=math.pi/4, omega_Y=math.pi/4, max_iter=10, lam=1, device="cuda"):
    embedding_dim_X = X.size(1)
    pro_X = rand_slices(embedding_dim_X, num_slices).to(device)
    embedding_dim_Y = Y.size(1)
    pro_Y = rand_slices(embedding_dim_Y, num_slices).to(device)
    X_detach = X.detach()
    Y_detach = Y.detach()
    for _ in range(max_iter):
        projections = f1(pro_X,pro_Y)
        arccos = arccos_distance_torch(projections, projections)
        reg = -lam * (arccos-omega_X)
        encoded_projections_X = X_detach.matmul(projections.transpose(0, 1))
        projections = f2(pro_X,pro_Y)
        arccos = arccos_distance_torch(projections, projections)
        reg += -lam * (arccos-omega_Y)
        encoded_projections_Y = Y_detach.matmul(projections.transpose(0, 1))

        MI=I_est(encoded_projections_X,encoded_projections_Y)

        # distribution_projections = Y_detach.matmul(projections.transpose(0, 1))
        # wasserstein_distance = torch.abs(
        #     (
        #         torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
        #         - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
        #     )
        # )
        # wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
        # wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)


        loss = reg - MI
        f1_op.zero_grad()
        f2_op.zero_grad()
        loss.backward(retain_graph=True)
        f1_op.step()
        f2_op.step()

    projections_X = f1(pro_X,pro_Y)
    projections_Y = f2(pro_X,pro_Y)
    encoded_projections_X = X.matmul(projections_X.transpose(0, 1))
    encoded_projections_Y = Y.matmul(projections_Y.transpose(0, 1))

    # distribution_projections = Y.matmul(projections.transpose(0, 1))
    # wasserstein_distance = torch.abs(
    #     (
    #         torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
    #         - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
    #     )
    # )
    # wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    # wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)

    MI=I_est(encoded_projections_X,encoded_projections_Y)
    

    return MI