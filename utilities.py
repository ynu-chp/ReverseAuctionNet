import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import torch.optim as optim

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def utility(batch_data, allocation, pay):
    """ Given input valuation , payment  and allocation , computes utility
            Input params:
                valuation : [num_batches, num_agents, num_items]
                allocation: [num_batches, num_agents, num_items]
                pay       : [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
    """
    return (pay-torch.sum(batch_data[0] * allocation, dim=-1)  )#(bs,n)





def misreportUtility(mechanism, batch_data, batchMisreports):
    """ This function takes the valuation and misreport batches
        and returns a tensor constaining all the misreported utilities

        #batchMisreports.shape torch.Size([500, 1, 2, 6])
        batchMisreports.shape torch.Size([500, 100, 2, 6])
    """

    batchTrue_bid= batch_data[0]
    batchTrue_value = batch_data[1]

    nAgent = batchTrue_bid.shape[-2]
    nObjects = batchTrue_bid.shape[-1]
    batchSize = batchTrue_bid.shape[0]
    nbrInitializations = batchMisreports.shape[1]


    V = batchTrue_bid.unsqueeze(1)  # (bs, 1, n_bidder, n_item)
    V = V.repeat(1, nbrInitializations, 1, 1)  # (bs, n_init, n_bidder, n_item)
    V = V.unsqueeze(0)  # (1, bs, n_init, n_bidder, n_item)
    V = V.repeat(nAgent, 1, 1, 1, 1)  # (n_bidder, bs, n_init, n_bidder, n_item)

    W = batchTrue_value.unsqueeze(1)
    W = W.repeat(1, nbrInitializations, 1, 1)
    W = W.unsqueeze(0)
    W = W.repeat(nAgent, 1, 1, 1, 1)

    M = batchMisreports.unsqueeze(0)
    M = M.repeat(nAgent, 1, 1, 1, 1)  # (n_bidder, bs, n_init, n_bidder, n_item)

    mask1 = np.zeros((nAgent, nAgent, nObjects))
    mask1[np.arange(nAgent), np.arange(nAgent), :] = 1.0
    mask2 = np.ones((nAgent, nAgent, nObjects))
    mask2 = mask2 - mask1

    mask1 = (torch.tensor(mask1).float()).to(device)
    mask2 = (torch.tensor(mask2).float()).to(device)

    V = V.permute(1, 2, 0, 3, 4)  # (bs, n_init, n_bidder, n_bidder, n_item)
    W = W.permute(1, 2, 0, 3, 4)
    M = M.permute(1, 2, 0, 3, 4)  # (bs, n_init, n_bidder, n_bidder, n_item)



    tensor = M * mask1 + V * mask2

    tensor = tensor.permute(2, 0, 1, 3, 4)  # (n_bidder, bs, n_init, n_bidder, n_item)
    W = W.permute(2, 0, 1, 3, 4)



    tensor = View(-1, nAgent, nObjects)(tensor)  # (n_bidder * bs * n_init, n_bidder, n_item)
    W = View(-1, nAgent, nObjects)(W)
    tensor = tensor.float()

    allocation, payment = mechanism(tensor, W)  # (n_bidder * bs * n_init, n_bidder, n_item)

    V = V.permute(2, 0, 1, 3, 4)  # (n_bidder, bs, n_init, n_bidder, n_item)
    M = M.permute(2, 0, 1, 3, 4)  # (n_bidder, bs, n_init, n_bidder, n_item)

    allocation = View(nAgent, batchSize, nbrInitializations, nAgent, nObjects)(allocation)
    payment = View(nAgent, batchSize, nbrInitializations, nAgent)(payment)


    advUtilities = payment- torch.sum(allocation * V, dim=-1)

    advUtility = advUtilities[np.arange(nAgent), :, :, np.arange(nAgent)]

    return (advUtility.permute(1, 2, 0))




def misreportOptimization(mechanism, batch, data, misreports, R, gamma):

    localMisreports = misreports[:]

    batchMisreports = torch.tensor(misreports[batch]).to(device)

    batchTrue_bid = torch.tensor(data[0][batch]).to(device)
    batchTrue_value=torch.tensor(data[1][batch]).to(device)


    batch_data = (batchTrue_bid, batchTrue_value)

    batchMisreports.requires_grad = True

    opt = optim.Adam([batchMisreports], lr=gamma)

    for k in range(R):
        advU = misreportUtility(mechanism, batch_data, batchMisreports)
        loss = -1 * torch.sum(advU).to(device)


        opt.zero_grad()
        loss.backward()
        opt.step()

    mechanism.zero_grad()

    localMisreports[batch, :, :, :] = batchMisreports.cpu().detach().numpy()
    return (localMisreports)


def trueUtility(batch_data, allocation=None, payment=None):
    """ This function takes the valuation batches
        and returns a tensor constaining the utilities
    """
    return utility(batch_data, allocation, payment)


def regret(mechanism, batch_data, batchMisreports, allocation, payment):
    """ This function takes the valuation and misreport batches
        and returns a tensor constaining the regrets for each bidder and each batch


    """
    missReportUtilityAll = misreportUtility(mechanism, batch_data, batchMisreports)
    misReportUtilityMax = torch.max(missReportUtilityAll, dim=1)[0]#(bs,n)
    return (misReportUtilityMax - trueUtility(batch_data, allocation, payment))



def loss(mechanism, lamb_r, rho_r, lamb_i, rho_i,lamb_b, rho_b, batch, data, misreports,budget):

    """
    This function tackes a batch which is a numpy array of indices and computes
    the loss function                                                             : loss
    the average regret per agent which is a tensor of size [nAgent]               : rMean
    the maximum regret among all batches and agenrs which is a tensor of size [1] : rMax
    the average payments which is a tensor of size [1]                            : -paymentLoss

    """
    batchMisreports = torch.tensor(misreports[batch]).to(device)

    batchTrue_bid = torch.tensor(data[0][batch]).to(device)
    batchTrue_value = torch.tensor(data[1][batch]).to(device)

    allocation, payment = mechanism(batchTrue_bid,batchTrue_value)
    paymentLoss = -torch.sum(allocation*batchTrue_value) / batch.shape[0]

    batch_data = (batchTrue_bid, batchTrue_value)
    r = F.relu(regret(mechanism, batch_data, batchMisreports, allocation, payment))

    # rgt
    rMean = torch.mean(r, dim=0).to(device)
    rMax = torch.max(r).to(device)
    lagrangianLoss_r = torch.sum(rMean * lamb_r)
    lagLoss_r = (rho_r / 2) * torch.sum(torch.pow(rMean, 2))

    # Ir
    I=F.relu(torch.sum(allocation*batchTrue_bid,dim=-1)-payment)# (bs,n)
    I_Mean=torch.mean(I, dim=0).to(device)#(n,)
    I_Max=torch.max(I).to(device) #(1,)
    lagrangianLoss_I= torch.sum(I_Mean*lamb_i)
    lagLoss_I = (rho_i / 2) * torch.sum(torch.pow(I_Mean, 2))

    # Bf
    B=F.relu(torch.sum(payment,dim=1)-budget)#
    B_Mean = (torch.sum(B) /batch.shape[0]).to(device)
    B_Max=torch.max(B).to(device)
    lagrangianLoss_B= torch.sum(B_Mean*lamb_b).to(device)
    lagLoss_B= (rho_b / 2) * torch.sum(torch.pow(B_Mean, 2)).to(device)


    loss = paymentLoss + lagrangianLoss_r + lagLoss_r + lagrangianLoss_I + lagLoss_I + lagrangianLoss_B + lagLoss_B
    return (loss, rMean, rMax,I_Mean,I_Max,B_Mean,B_Max, -paymentLoss)





