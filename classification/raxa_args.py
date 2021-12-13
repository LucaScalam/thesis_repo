#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: lucascalam
"""
import argparse


# Argumentos por linea de comandos
parser = argparse.ArgumentParser()
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=1e-3,
    help="Learning rate (default: 1e-3)",
)
parser.add_argument(
    "-rf",
    "--regularizer_factor",
    type=float,
    default=0,
    help="Regularizer factor (default: 0)",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=200,
    help="Epochs (default: 200)",
)
parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=16,
    help="Batch size (default: 16)",
)
parser.add_argument(
    "-do",
    "--Dropout",
    type=float,
    default=0,
    help="Dropout argument (default: 0)",
)
parser.add_argument(
    "-nn",
    "--NumNeuronas",
    type=int,
    default=16,
    help="Numero de neuronas (default: 16)",
)
parser.add_argument(
    "-swa_opt",
    "--SWA_Optim",
    type=int,
    default=0,
    help="SWA optimizer for training (default: False)",
)
parser.add_argument(
    "-idx",
    "--model_index",
    type=int,
    default=0,
    help="model_index (default: 0)",
)
parser.add_argument(
    "-nmod",
    "--number_mod",
    type=int,
    default=1,
    help="number of models to be trained (default: 1)",
)
parser.add_argument(
    "-pmod",
    "--prev_mod",
    type=int,
    default=0,
    help="to use trained model or not (default: 0)",
)
parser.add_argument(
    "-td2",
    "--train_data_2",
    type=int,
    default=0,
    help="to use train_data_2 or not (default: False)",
)

parser.add_argument(
    "-cmp",
    "--campaign",
    type=int,
    default=1819,
    help="1819 or 1920 for the campaign to be used (default: 1819)",
)

parser.add_argument(
    "-coef",
    "--coefficient",
    type=int,
    default=0,
    help="if the dataset has a coefficient inside bands field, use '1' (default: 0)",
)

kwargs = vars(parser.parse_args())
lr = kwargs["learning_rate"]
rf = kwargs["regularizer_factor"]
epochs = kwargs['epochs']
batch_size = kwargs['batch_size']
drop_arg = kwargs['Dropout']
nn = kwargs['NumNeuronas']
swa_opt = kwargs['SWA_Optim']
idx = kwargs['model_index']
nmod = kwargs['number_mod']
pmod = kwargs['prev_mod']
td2 = kwargs['train_data_2']
cmp = kwargs['campaign']
coef = kwargs['coefficient']

print("-----------------------------------------------------")
print('lr: {} rf: {} do: {} epochs: {} bs: {} nn: {} swa_opt: {} idx: {} nmod: {} pmod: {} td2: {} cmp: {} coef: {}'.format(
    lr, rf, drop_arg, epochs, batch_size, nn, swa_opt, idx, nmod, pmod, td2, cmp, coef))
print("-----------------------------------------------------")

