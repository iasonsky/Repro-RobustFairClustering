import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import random
import kmedoids
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from zoopt import Dimension, ValueType, Objective, Parameter, Opt, ExpOpt
import warnings 
warnings.filterwarnings('ignore')
import numpy as np
from fair_clustering.eval.functions import * #[TO-DO] Write base class and derive metrics from it, temporary eval code
from fair_clustering.dataset import ExtendedYaleB, Office31, MNISTUSPS
from fair_clustering.algorithm import FairSpectral, FairKCenter, FairletDecomposition, ScalableFairletDecomposition
import argparse
import pickle

# Set parameters related to dataset and get dataset
parser = argparse.ArgumentParser(description='Fair Clustering')
parser.add_argument('--dataset', type=str, default='Office-31', metavar='N',
                    help='dataset to use')
parser.add_argument('--cl_algo', type=str, default='SFD', metavar='N',
                    help='clustering algorithm to use')
name = parser.parse_args().dataset
cl_algo = parser.parse_args().cl_algo

#Choose between Office-31, MNIST_USPS, Yale, or DIGITS
if name == 'Office-31':
  dataset = Office31(exclude_domain='amazon', use_feature=True)
  X, y, s = dataset.data
elif name == 'MNIST_USPS':
  dataset = MNISTUSPS(download=True)
  X, y, s = dataset.data
elif name == 'Yale':
  dataset = ExtendedYaleB(download=True, resize=True)
  X, y, s = dataset.data
elif name == 'DIGITS':
  X, y, s = np.load('X_' + name + '.npy'), np.load('y_' + name + '.npy'), np.load('s_' + name + '.npy')
else:
  print('Invalid dataset name')
  sys.exit()

# cl_algo can only be FSC or SFD #TODO: Add KFC
if cl_algo != 'FSC' and cl_algo != 'SFD':
  print('Invalid clustering algorithm name')
  sys.exit()


# Fairness Attack
def attack_balance(solution):
  X_copy, s_copy = X.copy(), s.copy()
  flipped_labels = solution.get_x()
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1
  fair_clustering_algo = select_clustering_algorithm(name, cl_algo, n_clusters, random_state)
  fair_clustering_algo.fit(X_copy, s_copy)
  labels_sfd = fair_clustering_algo.labels_

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

  return bal


def attack_entropy(solution):
  X_copy, s_copy = X.copy(), s.copy()
  flipped_labels = solution.get_x()
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1
  fair_clustering_algo = select_clustering_algorithm(name, cl_algo, n_clusters, random_state)
  fair_clustering_algo.fit(X_copy, s_copy)
  labels_sfd = fair_clustering_algo.labels_

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

  ent = entropy(labels_sfd_eval, s_eval)

  return ent

def process_solution(sol):
  X_copy, s_copy, y_copy = X.copy(), s.copy(), y.copy()
  flipped_labels = sol.get_x()
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1
  fair_clustering_algo = select_clustering_algorithm(name, cl_algo, n_clusters, random_state)
  fair_clustering_algo.fit(X_copy, s_copy)
  labels_sfd = fair_clustering_algo.labels_

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

def conduct_random_attack(size_sol):
  X_copy, s_copy, y_copy = X.copy(), s.copy(), y.copy()
  random.seed(None)
  flipped_labels = [random.randint(0,1) for _ in range(size_sol)]
  i = 0
  for idx in U_idx:
    s_copy[idx] = flipped_labels[i]
    i += 1
  fair_clustering_algo = select_clustering_algorithm(name, cl_algo, n_clusters, random_state)
  fair_clustering_algo.fit(X_copy, s_copy)
  labels_sfd = fair_clustering_algo.labels_

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

def calculate_budget(name, cl_algo):
  '''10 for FSC for MNIST_USPS and 50 for SFD for MNIST_USPS,
     20 for FSC for Office-31 and 20 for SFD for Office-31, 
     10 for FSC for Yale and 20 for SFD for Yale, 
     15 for FSC for DIGITS and 25 for SFD for DIGITS'''
  if name == 'MNIST_USPS':
    if cl_algo == 'FSC':
      return 10
    elif cl_algo == 'SFD':
      return 50
  elif name == 'Office-31':
    return 20
  elif name == 'Yale':
    if cl_algo == 'FSC':
      return 10
    elif cl_algo == 'SFD':
      return 20
  elif name == 'DIGITS':
    if cl_algo == 'FSC':
      return 15
    elif cl_algo == 'SFD':
      return 25

def select_clustering_algorithm(name, cl_algo, n_clusters, random_state): #TODO: Add KFC
  '''Selects the clustering algorithm based on the dataset and the clustering algorithm name'''
  metric_str = 'manhattan' if name == 'MNIST_USPS' else 'euclidean'
  if cl_algo == 'FSC':
      return FairSpectral(n_clusters=n_clusters, num_neighbors=3, metric_str=metric_str, random_state=random_state)
  elif cl_algo == 'SFD':
      beta = 1 if name == 'DIGITS' else 2
      return ScalableFairletDecomposition(n_clusters=n_clusters, alpha=5, beta=beta, random_state=random_state)

def print_and_save_results(result_dict, result_name, name, cl_algo):
    '''Prints and saves the results in a pickle file.'''
    print(f'{result_name} Results')
    for metric in result_dict.keys():
        mean_val = np.mean(result_dict[metric])
        std_val = np.std(result_dict[metric])
        print(f'{metric}: Mean = {mean_val}, Std = {std_val}')

    # Save results to a pickle file
    with open(f'{result_name.lower()}_res_{name}_{cl_algo}.pkl', 'wb') as file:
        pickle.dump(result_dict, file)

def create_objective(name, cl_algo, dim_size, attack_balance, attack_entropy):
    dim = Dimension(dim_size, [[0, 1]]*dim_size, [False]*dim_size)

    if name == 'Office-31':
        return Objective(attack_balance, dim)
    elif name in ['MNIST_USPS', 'DIGITS']:
        if cl_algo == 'SFD':
            return Objective(attack_balance, dim)
        elif cl_algo == 'FSC':
            return Objective(attack_entropy, dim)
    elif name == 'Yale':
        return Objective(attack_entropy, dim)
    else:
        raise ValueError(f"Unrecognized dataset or clustering algorithm: {name}, {cl_algo}")

# Main code
n_clusters = len(np.unique(y))
print(f"{X.shape}, {y.shape}, {s.shape}")
print(f"Clustering Algorithm: {cl_algo}")
print(f"Dataset: {name}")
print(f"Number of Clusters: {n_clusters}")
print(f"Number of Data Points: {len(y)}")
print(f"Number of Sensitive Features: {len(np.unique(s))}")
print(f"Number of Features: {X.shape[1]}")
print(f"# of clusters -> {n_clusters}")

seeds = [150, 1, 4200, 424242, 1947, 355, 256, 7500, 99999, 18]
n_trials = len(seeds)

U_idx_full, V_idx_full = np.load('U_idx_' + name + '.npy').tolist(), np.load('V_idx_' + name + '.npy').tolist()

# Calculate the 15% index
j = int(0.15 * len(U_idx_full))

# Initialize result dictionaries
pre_attack_res = {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []}
post_attack_res = {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []}
random_attack_res = {'BALANCE': [], 'ENTROPY': [], 'ACC': [], 'NMI': []}

# Update the indices for the current percentage
U_idx = U_idx_full[:j]
V_idx = V_idx_full

for trial_idx in range(n_trials):
  random_state = seeds[trial_idx]

  # Initialize clustering algorithm
  fair_algo = select_clustering_algorithm(name, cl_algo, n_clusters, random_state)
  fair_algo.fit(X, s)
  labels = fair_algo.labels_

  s_test = np.array([s[idx] for idx in V_idx])
  X_test = np.array([X[idx] for idx in V_idx])
  labels_test = np.array([labels[idx] for idx in V_idx])
  y_test = np.array([y[idx] for idx in V_idx])
  
  # Store pre-attack results
  pre_attack_res['BALANCE'].append(balance(labels_test, X_test, s_test))
  pre_attack_res['ENTROPY'].append(entropy(labels_test, s_test))
  pre_attack_res['ACC'].append(acc(y_test, labels_test))
  pre_attack_res['NMI'].append(nmi(y_test, labels_test))

  dim_size = len(U_idx)
  obj = create_objective(name, cl_algo, dim_size, attack_balance, attack_entropy)  
  budget = calculate_budget(name, cl_algo)
  solution = Opt.min(obj, Parameter(budget=budget)) 
  
  pa_bal, pa_ent, pa_acc, pa_nmi = process_solution(solution)
  post_attack_res['BALANCE'].append(pa_bal)
  post_attack_res['ENTROPY'].append(pa_ent)
  post_attack_res['ACC'].append(pa_acc)
  post_attack_res['NMI'].append(pa_nmi)

  r_bal, r_ent, r_acc, r_nmi = conduct_random_attack(dim_size)
  random_attack_res['BALANCE'].append(r_bal)
  random_attack_res['ENTROPY'].append(r_ent)
  random_attack_res['ACC'].append(r_acc)
  random_attack_res['NMI'].append(r_nmi)

print_and_save_results(pre_attack_res, 'Pre-Attack', name, cl_algo)
print_and_save_results(post_attack_res, 'Post-Attack', name, cl_algo)
print_and_save_results(random_attack_res, 'Random Attack', name, cl_algo)