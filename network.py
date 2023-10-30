import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Transformer2DNet(nn.Module):
    def __init__(self, d_input, d_output, n_layer, n_head):

        super(Transformer2DNet, self).__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.n_layer = n_layer

        d_in = d_input
        d_hidden=4*d_in

        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(n_layer):
            d_out = d_in if i != n_layer - 1 else d_output

            self.row_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))
            self.col_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))
            self.fc.append(nn.Sequential(
                nn.Linear(2 * d_in, d_in),
                nn.ReLU(),
                nn.Linear(d_in, d_out)
            ))


    def forward(self, input):
        bs, n_bidder, n_item, d = input.shape
        x = input
        for i in range(self.n_layer):
            row_x = x.view(-1, n_item, d)
            row = self.row_transformer[i](row_x)
            row = row.view(bs, n_bidder, n_item, -1)

            col_x = x.permute(0, 2, 1, 3).reshape(-1, n_bidder, d)
            col = self.col_transformer[i](col_x)
            col = col.view(bs, n_item, n_bidder, -1).permute(0, 2, 1, 3)



            x = torch.cat([row, col], dim=-1)

            x = self.fc[i](x)
        return x

class TransformerMechanism(nn.Module):
    def __init__(self,  n_layer, n_head, d_hidden):
        super(TransformerMechanism, self).__init__()

        self.pre_net1 = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden-3)
        )

        self.pre_net2 = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden-3)
        )

        d_input = 2*d_hidden
        self.n_layer, self.n_head  =  n_layer, n_head
        self.mechanism = Transformer2DNet(d_input, 2, self.n_layer, self.n_head)


    def forward(self, batch_bid,batch_value):
        bid, value = batch_bid,batch_value

        x1 = bid.unsqueeze(-1)
        x2 = value.unsqueeze(-1)

        x3=self.pre_net1(x1)
        x4=self.pre_net2(x2)

        y1=torch.mean(bid,dim=1,keepdim=True)
        y1=y1.repeat(1,bid.shape[1],1).unsqueeze(-1)
        y2=torch.mean(bid,dim=2,keepdim=True)
        y2= y2.repeat(1, 1,bid.shape[2]).unsqueeze(-1)

        y3=torch.mean(value,dim=1,keepdim=True)
        y3=y3.repeat(1,value.shape[1],1).unsqueeze(-1)
        y4=torch.mean(value,dim=2,keepdim=True)
        y4= y4.repeat(1, 1,value.shape[2]).unsqueeze(-1)

        x = torch.cat([x1,y1,y2,x3 , x2 ,y3,y4, x4], dim=-1)


        mechanism = self.mechanism(x)
        allocation, payment = \
            mechanism[:, :, :, 0], mechanism[:, :, :, 1]


        allocation=torch.sigmoid(allocation)

        payment = payment.mean(-1)


        return allocation, payment


