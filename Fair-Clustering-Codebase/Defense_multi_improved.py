#!/usr/bin/env python
# coding: utf-8

# In[1]:
from __future__ import division
from __future__ import print_function
import subprocess


subprocess.call(['pip', 'install', 'kmedoids'])
subprocess.call(['pip', 'install', 'gdown'])
subprocess.call(['pip', 'install', 'python-mnist'])
subprocess.call(['pip', 'install', 'pulp'])
subprocess.call(['pip', 'install', 'zoopt'])
subprocess.call(['pip', 'install', 'pyckmeans'])
subprocess.call(['pip', 'install', 'scikit-learn==1.2.2', '--upgrade'])


import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' #:4096:8


import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import random
import kmedoids
from sklearn.decomposition import PCA
from zoopt import Dimension, ValueType, Objective, Parameter, Opt, ExpOpt
import seaborn as sns
import subprocess
import torch

import warnings 
warnings.filterwarnings('ignore')

from fair_clustering.eval.functions import * #[TO-DO] Write base class and derive metrics from it, temporary eval code

from fair_clustering.dataset import ExtendedYaleB, Office31, MNISTUSPS, ExtendedYaleB_alter
from fair_clustering.algorithm import FairSpectral, FairKCenter, FairletDecomposition, ScalableFairletDecomposition
from holisticai.bias.metrics import cluster_balance, cluster_dist_entropy, cluster_dist_kl, cluster_dist_l1, silhouette_diff, min_cluster_ratio

import matplotlib.pyplot as plt


# Set parameters related to dataset and get dataset

name = 'Office-31' #Choose between Office-31, MNIST_USPS, Yale, or DIGITS

if name == 'Office-31':
  dataset = Office31(exclude_domain='amazon', use_feature=True)
  X, y, s = dataset.data
elif name == 'MNIST_USPS':
  dataset = MNISTUSPS(download=True)
  X, y, s = dataset.data
elif name == 'Yale':
  dataset = ExtendedYaleB(download=True, resize=True)
  X, y, s = dataset.data
elif name == 'Yale_alter':
  dataset = ExtendedYaleB_alter(resize=True)
  X, y, s = dataset.data
  print(f"y:", np.unique(y))
elif name == 'DIGITS':
  X, y, s = np.load('X_' + name + '.npy'), np.load('y_' + name + '.npy'), np.load('s_' + name + '.npy')

print(X.shape, y.shape, s.shape)
print("unique y_in:", np.unique(y))

# Fairness Defense
from pyckmeans import CKmeans

import random
import time
import argparse
import numpy as np

L = np.load('Consensus-Fair-Clustering/precomputed_labels/labels_' + name + '.npy')
U_index = np.load('U_idx_Yale' + '.npy')
V_index = np.load('V_idx_Yale' + '.npy')
#print(f"L: ",L[-100:])
print(len(L))
print(f"U_idx: ", U_index)
print(f"V_idx: ", V_index)

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from models import GMLP, ClusteringLayer
from utils import get_A_r, sparse_mx_to_torch_sparse_tensor, target_distribution, aff

from scipy import sparse
from torch import nn


def Ncontrast(x_dis, adj_label, tau = 1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_batch(batch_size, idx_train, adj_label, features):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long).cuda()
    rand_indx[0:len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx,:][:,rand_indx]
    return features_batch, adj_label_batch

def train(model, CL, optimizer, s_idx0, s_idx1, bs, KL_div, tau, alpha, beta, idx_train, adj_label, features, Y, MSEL):
    features_batch, adj_label_batch = get_batch(bs, idx_train, adj_label, features)
    model.train()
    CL.train()

    optimizer.zero_grad()
    output, x_dis, embeddings = model(features_batch)
    
    output = CL(embeddings)
    output0, output1 = output[s_idx0], output[s_idx1]
    target0, target1 = target_distribution(output0).detach(), target_distribution(output1).detach()
    fair_loss = 0.5 * KL_div(output0.log(), target0) + 0.5 * KL_div(output1.log(), target1)

    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau = tau)

    predict0, predict1 = Y[s_idx0], Y[s_idx1]
    partition_loss = 0.5 * MSEL(aff(output0), aff(predict0)) + 0.5 * MSEL(aff(output1), aff(predict1))

    loss_train = alpha * fair_loss + loss_Ncontrast + beta * partition_loss

    loss_train.backward()
    optimizer.step()
    return


def ConsensusFairClusteringHelper(name, X_in, s_in, y_in, save, order=1, lr=0.01, weight_decay=5e-3, alpha=50.0, num_hidden=256, bs=3800, tau=2, epochs=3000, dropout=0.6):
  k = len(np.unique(y_in))

  if name == 'Office-31':
    beta = 100.0 
    alpha = 1.0 
    order = 1
  if name == 'MNIST_USPS':
    beta = 25.0 
    alpha = 100.0 
    order = 2
  if name == 'Yale':
    beta = 10.0 
    alpha = 50.0 
    order = 2
  if name == 'Yale_alter':
    beta = 10.0
    alpha = 50.0
    order = 2
  if name == 'DIGITS':
    beta = 50.0 
    alpha = 10.0 
    order = 2
    num_hidden=36


  ckm = CKmeans(k=k, n_rep=100, p_samp=0.5, p_feat=0.5, random_state=42)
  ckm.fit(X_in)
  ckm_res = ckm.predict(X_in, return_cls=True)


  adj, features, labels = ckm_res.cmatrix, X_in, y_in
  adj = sparse.csr_matrix(adj)
  adj = sparse_mx_to_torch_sparse_tensor(adj).float()
  features = torch.FloatTensor(features).float()
  labels = torch.LongTensor(labels)
  idx_train = np.array(range(len(features)))
  idx_train = torch.LongTensor(idx_train)

  adj_label = get_A_r(adj, order)
  adj, adj_label, features, idx_train = adj.cuda(), adj_label.cuda(), features.cuda(), idx_train.cuda()

  s_idx0, s_idx1 = [], []
  for i in range(len(s_in)):
    if s_in[i] == 0:
      s_idx0.append(i)
    elif s_in[i] == 1:
      s_idx1.append(i) 


  L = np.load('Consensus-Fair-Clustering/precomputed_labels/labels_' + name + '.npy')
  Y = np.zeros((len(s), k))
  for i,l in enumerate(L):
    Y[i,l] = 1.0
  Y = torch.FloatTensor(Y).float().cuda()
  MSEL = nn.MSELoss(reduction="sum")

  torch.manual_seed(42)
  torch.use_deterministic_algorithms(True)
  model = GMLP(nfeat=features.shape[1],
              nhid=num_hidden,
              nclass=labels.max().item() + 1,
              dropout=dropout,
              )

  torch.manual_seed(42)
  torch.use_deterministic_algorithms(True)
  CL = ClusteringLayer(cluster_number=k, hidden_dimension=num_hidden).cuda()
  
  optimizer = optim.Adam(model.get_parameters() + CL.get_parameters(), lr=lr, weight_decay=weight_decay)
  KL_div = nn.KLDivLoss(reduction="sum")
  model.cuda()
  features = features.cuda()
  labels = labels.cuda()
  idx_train = idx_train.cuda()

  for epoch in tqdm(range(epochs)):
    train(model, CL, optimizer, s_idx0, s_idx1, bs, KL_div, tau, alpha, beta, idx_train, adj_label, features, Y, MSEL)

  model.eval()
  logits, embeddings = model(features)
  CL.eval()
  preds = CL(embeddings)
  preds = preds.cpu().detach().numpy()
  pred_labels = np.argmax(preds, axis=1)

  return pred_labels


def ConsensusFairClustering(name, X_in, s_in, y_in, save):
  name_bal = {'Office-31': 0.5, 'MNIST_USPS': 0.3, 'DIGITS': 0.1, 'Yale': 0.1, 'Yale_alter': 0.1}
  name_ent = {'Office-31': 5.5, 'MNIST_USPS': 2.2, 'DIGITS': 1.6, 'Yale': 5.2, 'Yale_alter': 5}
  while True: #Sometimes the model optimizes for a local minima which is why we can run enough times to get a good representation learnt
    cfc_labels = ConsensusFairClusteringHelper(name, X_in, s_in, y_in, save)
    # threshold -> 0.5 for Office-31 and 0.3 (0.4) for MNIST_USPS and 0.1 for DIGITS and 0.1 for Yale
    if balance(cfc_labels, X_in, s_in) >= name_bal[name] and entropy(cfc_labels, s_in) >= name_ent[name]:
      break
  print("\nCompleted CFC model training.")
  return cfc_labels


# Trial run!
lbls = ConsensusFairClustering(name, X, s, y, save=False)
print(lbls)


# Check to see metrics too!
print("balance: {}".format(balance(lbls, X, s)))
print("entropy: {}".format(entropy(lbls, s)))
print("nmi: {}".format(nmi(y, lbls)))
print("acc: {}".format(acc(y, lbls)))


def attack_balance(solution):
  X_copy, s_copy = X.copy(), s.copy()
  flipped_labels = solution.get_x()
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1

  labels_sfd = ConsensusFairClustering(name, X_copy, s_copy, y, save=False)
  
  s_eval = []
  X_eval = []
  labels_sfd_eval = []
  for idx in V_idx:
    s_eval.append(s_copy[idx])
    X_eval.append(X_copy[idx])
    labels_sfd_eval.append(labels_sfd[idx])
  s_eval = np.array(s_eval)
  X_eval = np.array(X_eval)
  labels_sfd_eval = np.array(labels_sfd_eval)

  bal = balance(labels_sfd_eval, X_eval, s_eval)
  ent = entropy(labels_sfd_eval, s_eval)

  return bal


def attack_entropy(solution):
  X_copy, s_copy = X.copy(), s.copy()
  flipped_labels = solution.get_x()
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1

  labels_sfd = ConsensusFairClustering(name, X_copy, s_copy, y, save=False)

  s_eval = []
  X_eval = []
  labels_sfd_eval = []
  for idx in V_idx:
    s_eval.append(s_copy[idx])
    X_eval.append(X_copy[idx])
    labels_sfd_eval.append(labels_sfd[idx])
  s_eval = np.array(s_eval)
  X_eval = np.array(X_eval)
  labels_sfd_eval = np.array(labels_sfd_eval)

  bal = balance(labels_sfd_eval, X_eval, s_eval)
  ent = entropy(labels_sfd_eval, s_eval)

  return ent


def process_solution(sol):
  X_copy, s_copy, y_copy = X.copy(), s.copy(), y.copy()
  flipped_labels = sol.get_x()
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1

  labels_sfd = ConsensusFairClustering(name, X_copy, s_copy, y, save=False)

  s_eval = []
  X_eval = []
  labels_sfd_eval = []
  y_eval = []
  for idx in V_idx:
    s_eval.append(s_copy[idx])
    X_eval.append(X_copy[idx])
    labels_sfd_eval.append(labels_sfd[idx])
    y_eval.append(y_copy[idx])
  s_eval = np.array(s_eval)
  X_eval = np.array(X_eval)
  labels_sfd_eval = np.array(labels_sfd_eval)
  y_eval = np.array(y_eval)

  bal = balance(labels_sfd_eval, X_eval, s_eval)
  ent = entropy(labels_sfd_eval, s_eval)
  accuracy = acc(y_eval, labels_sfd_eval)
  nmi_score = nmi(y_eval, labels_sfd_eval)

  return (bal, ent, accuracy, nmi_score)


n_clusters = len(np.unique(y))
print("# of clusters -> " + str(n_clusters))
n_trials = 1

U_idx_full, V_idx_full = np.load('U_idx_' + name + '.npy').tolist(), np.load('V_idx_' + name + '.npy').tolist()

cfc_pre_res = {
    0 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
}


cfc_post_res = {
    0 : {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []},
}

for percent, j in enumerate([int(0.15*len(U_idx_full))]):

  U_idx = U_idx_full[:j]
  V_idx = V_idx_full

  for trial_idx in range(n_trials):

    labels = ConsensusFairClustering(name, X, s, y, save=False)

    s_test = []
    X_test = []
    labels_test = []
    y_test = []
    for idx in V_idx:
      s_test.append(s[idx])
      X_test.append(X[idx])
      labels_test.append(labels[idx])
      y_test.append(y[idx])
    s_test = np.array(s_test)
    X_test = np.array(X_test)
    labels_test = np.array(labels_test)
    y_test = np.array(y_test)

    cfc_pre_res[percent]['BALANCE'].append(balance(labels_test, X_test, s_test))
    cfc_pre_res[percent]['ENTROPY'].append(entropy(labels_test, s_test))
    cfc_pre_res[percent]['ACC'].append(acc(y_test, labels_test))
    cfc_pre_res[percent]['NMI'].append(nmi(y_test, labels_test))

    dim_size = len(U_idx)
    dim = Dimension(dim_size, [[0, 1]]*dim_size, [False]*dim_size)
    obj = Objective(attack_entropy, dim)
    solution = Opt.min(obj, Parameter(budget=5)) 

    pa_bal, pa_ent, pa_acc, pa_nmi = process_solution(solution)

    cfc_post_res[percent]['BALANCE'].append(pa_bal)
    cfc_post_res[percent]['ENTROPY'].append(pa_ent)
    cfc_post_res[percent]['ACC'].append(pa_acc)
    cfc_post_res[percent]['NMI'].append(pa_nmi)

print(f"dataset: ", name)
print(f"pre-res: ", cfc_pre_res)
print(f"post-res: ", cfc_post_res)

