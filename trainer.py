import os
import sys
import numpy as np
from statistics import mean
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import shutil
from time import time

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
datefmt="%Y-%m-%d,%H:%M:%S",
)

from network import TransformerMechanism
from utilities import misreportOptimization
def loss_function(mechanism,lamb_r,rho_r,lamb_i,rho_i,lamb_b,rho_b,batch,trueValuations,misreports,budget):
    from utilities import loss
    return loss(mechanism,lamb_r,rho_r,lamb_i,rho_i,lamb_b,rho_b,batch,trueValuations,misreports,budget)


class Trainer():
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.set_data(args)
        self.set_model(args)

        self.rho_r = args.rho_r
        self.lamb_r = args.lamb_r * torch.ones(args.n_bidder).to(args.device)

        self.rho_i=args.rho_i
        self.lamb_i=args.lamb_i*torch.ones(args.n_bidder).to(args.device)

        self.lamb_b = args.lamb_b * torch.ones(1).to(args.device)
        self.rho_b=args.rho_b

        self.budget=args.budget
        self.n_iter = 0

    def set_data(self, args):
        def load_data(dir):
            data = [np.load(os.path.join(dir, 'bid.npy')).astype(np.float32),
                    np.load(os.path.join(dir, 'value.npy')).astype(np.float32)]

            return tuple(data)

        self.train_dir = os.path.join(args.data_dir, args.training_set)
        self.train_data = load_data(self.train_dir)
        self.train_data =tuple([self.train_data[0]/args.norm,self.train_data[1]/args.norm])
        self.train_size = len(self.train_data[0])

        self.misreports = np.random.uniform(self.train_data[0].min(), self.train_data[0].max(),size=(self.train_size, args.n_misreport_init_train,args.n_bidder, args.n_item))


        self.test_dir = os.path.join(args.data_dir, args.test_set)
        self.test_size = args.test_size
        self.test_data = load_data(self.test_dir)
        self.test_data = tuple([self.test_data[0] / args.norm, self.test_data[1] / args.norm])
        self.test_size = len(self.test_data[0])

    def set_model(self, args):
        self.mechanism = TransformerMechanism(args.n_layer, args.n_head, args.d_hidden).to(args.device)
        self.mechanism = nn.DataParallel(self.mechanism)
        # state_dict = torch.load('model/budget=8/2x6')
        # self.mechanism.load_state_dict(state_dict)
        self.optimizer = torch.optim.Adam(self.mechanism.parameters(), lr=args.learning_rate)


    def train(self, args):


        for epoch in range(args.n_epoch):
            profit_sum = 0
            regret_sum = 0
            regret_max = 0
            IR_sum=0
            IR_max=0
            Budget_sum=0
            Budget_max=0
            loss_sum = 0

            for i in tqdm(range(0, self.train_size, args.batch_size)):
                self.n_iter += 1
                batch_indices = np.random.choice(self.train_size, args.batch_size)


                self.misreports = misreportOptimization(self.mechanism, batch_indices, self.train_data, self.misreports,
                                                   args.r_train, args.gamma)



                loss, regret_mean_bidder, regret_max_batch, IR_mean_bidder,IR_max_batch,Budget_mean_bs,Budget_max_bs,profit = \
                    loss_function(self.mechanism, self.lamb_r, self.rho_r, self.lamb_i, self.rho_i, self.lamb_b,self.rho_b, batch_indices, self.train_data, self.misreports ,self.budget/args.norm)

                loss_sum += loss.item() * len(batch_indices)
                regret_sum += regret_mean_bidder.mean().item() * len(batch_indices)
                regret_max = max(regret_max, regret_max_batch.item())

                IR_sum += IR_mean_bidder.mean().item() * len(batch_indices)
                IR_max = max(IR_max, IR_max_batch.item())

                Budget_sum += Budget_mean_bs.mean().item()*len(batch_indices)
                Budget_max=max(Budget_max,Budget_max_bs.item())

                profit_sum += profit.item() * len(batch_indices)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                if self.n_iter % args.lamb_r_update_freq  == 0:
                    self.lamb_r += self.rho_r * regret_mean_bidder.detach()
                if self.n_iter % args.lamb_i_update_freq  == 0:
                    self.lamb_i += self.rho_i * IR_mean_bidder.detach()
                if self.n_iter % args.lamb_b_update_freq == 0:
                    self.lamb_b += self.rho_b * Budget_mean_bs.detach()

            if (epoch + 1) % args.rho_r_update_freq == 0:
                self.rho_r += args.delta_rho_r
            if (epoch + 1) % args.rho_i_update_freq == 0:
                self.rho_i += args.delta_rho_i
            if (epoch + 1) % args.rho_b_update_freq == 0:
                self.rho_b += args.delta_rho_b

            logging.info(f"Train: epoch={epoch + 1}, loss={loss_sum/self.train_size}, "f"profit={(profit_sum)/self.train_size}")

            logging.info(f"regret={(regret_sum) / self.train_size}, regret_max={regret_max}")
            logging.info(f"ir={(IR_sum) / self.train_size}, ir_max={IR_max}")
            logging.info(f"bf={(Budget_sum) / self.train_size}, bf_max={Budget_max}")


            logging.info(f"Train: rho_r={self.rho_r}, lamb_r={self.lamb_r.mean().item()} ")
            logging.info(f"Train: rho_i={self.rho_i}, lamb_i={self.lamb_i.mean().item()} ")
            logging.info(f"Train: rho_b={self.rho_b}, lamb_b={self.lamb_b.mean().item()} ")


            if (epoch + 1) % args.eval_freq== 0:
                self.test(args, valid=True)

        if not os.path.exists("model"):
            os.makedirs("model")
        if not os.path.exists(os.path.join("model", "budget="+str(args.budget))):
            os.makedirs(os.path.join("model", "budget="+str(args.budget)))

        model_path = os.path.join(os.path.join("model", "budget="+str(args.budget)), str(args.n_bidder) + "x6")
        torch.save(self.mechanism.state_dict(), model_path)

        logging.info('Final test')

        self.test(args)


    def test(self, args, valid=False, load=False):
        if valid:
            data_size = args.batch_test * 10
            indices = np.random.choice(self.test_size, data_size)
            data = tuple([x[indices] for x in self.test_data])


        else:
            data_size = self.test_size
            data = self.test_data


        misreports = np.random.uniform(data[0].min(), data[0].max(),
                                       size=(data_size, args.n_misreport_init, args.n_bidder, args.n_item))


        indices = np.arange(data_size)
        profit_sum = 0.0
        regret_sum = 0.0
        regret_max = 0.0
        IR_sum = 0
        IR_max = 0
        Budget_sum = 0
        Budget_max=0
        loss_sum = 0.0
        n_iter = 0.0
        for i in tqdm(range(0, data_size, args.batch_test)):
            batch_indices = indices[i:i+args.batch_test]

            n_iter += len(batch_indices)
            misreports = misreportOptimization(self.mechanism, batch_indices, data, misreports,
                                               args.r_test, args.gamma)

            with torch.no_grad():
                loss, regret_mean_bidder, regret_max_batch, IR_mean_bidder,IR_max_batch,Budget_mean_bs,Budget_max_bs,profit = \
                    loss_function(self.mechanism, self.lamb_r, self.rho_r, self.lamb_i, self.rho_i, self.lamb_b, self.rho_b,
                              batch_indices, data, misreports, self.budget / args.n_bid)
            loss_sum += loss.item() * len(batch_indices)
            regret_sum += regret_mean_bidder.mean().item() * len(batch_indices)
            regret_max = max(regret_max, regret_max_batch.item())

            IR_sum += IR_mean_bidder.mean().item() * len(batch_indices)
            IR_max = max(IR_max, IR_max_batch.item())

            Budget_sum += Budget_mean_bs.mean().item() * len(batch_indices)
            Budget_max=max(Budget_max,Budget_max_bs.item())

            profit_sum += profit.item() * len(batch_indices)

            if valid == False:
                logging.info(f"profit={(profit_sum)/n_iter}, regret={(regret_sum)/n_iter}, ir={(IR_sum)/n_iter}, bf={(Budget_sum)/n_iter}")

        logging.info(f"Test: loss={loss_sum/data_size}, profit={(profit_sum)/data_size}, "
                     f"regret={(regret_sum)/data_size}, regret_max={regret_max}, "
                     f"ir={(IR_sum)/data_size}, ir_max={IR_max}, "
                     f"bf={(Budget_sum)/data_size},bf_max={Budget_max}")






