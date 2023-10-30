import torch
import logging
from time import time
import datetime
from trainer import Trainer
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--data_dir', type=str, default='data/8x6')
    parser.add_argument('--training_set', type=str, default='train')
    parser.add_argument('--test_set', type=str, default='test')


    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_hidden', type=int, default=32)

    parser.add_argument('--n_bidder', type=int, default=8)
    parser.add_argument('--n_item', type=int, default=6)
    parser.add_argument('--r_train', type=int, default=25, help='Number of steps in the inner maximization loop')
    parser.add_argument('--r_test', type=int, default=200, help='Number of steps in the inner maximization loop when testing')
    parser.add_argument('--gamma', type=float, default=1e-3, help='The learning rate for the inner maximization loop')
    parser.add_argument('--norm', type=int, default=20)

    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--n_misreport_init', type=int, default=100)
    parser.add_argument('--n_misreport_init_train', type=int, default=1)

    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--n_epoch', type=int, default=80)
    parser.add_argument('--misreport_epoch', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--test_size', type=float, default=None)
    parser.add_argument('--batch_test', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=20)

    #Rgt
    parser.add_argument('--lamb_r', type=float, default=1)
    parser.add_argument('--lamb_r_update_freq', type=int, default=6)
    parser.add_argument('--rho_r', type=float, default=1)
    parser.add_argument('--rho_r_update_freq', type=int, default=3)
    parser.add_argument('--delta_rho_r', type=float, default=3)
    #Ir
    parser.add_argument('--lamb_i', type=float, default=1)
    parser.add_argument('--lamb_i_update_freq', type=int, default=6)
    parser.add_argument('--rho_i', type=float, default=1)
    parser.add_argument('--rho_i_update_freq', type=int, default=5)
    parser.add_argument('--delta_rho_i', type=float, default=1)
    #Bf
    parser.add_argument('--lamb_b', type=float, default=0.1)
    parser.add_argument('--lamb_b_update_freq', type=int, default=6)
    parser.add_argument('--rho_b', type=float, default=0.1)
    parser.add_argument('--rho_b_update_freq', type=int, default=6)
    parser.add_argument('--delta_rho_b', type=float, default=1)

    t0 = time()
    args = parser.parse_args()
    trainer = Trainer(args)

    trainer.train(args)

    time_used = time() - t0
    logging.info(f'Time Cost={datetime.timedelta(seconds=time_used)}')

