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
from fair_clustering.dataset import ExtendedYaleB, Office31, MNISTUSPS, MTFL_data, ExtendedYaleB_alter
from fair_clustering.algorithm import FairSpectral, FairKCenter, FairletDecomposition, ScalableFairletDecomposition
import argparse
import pickle
import os
from holisticai.bias.metrics import cluster_balance, cluster_dist_entropy, cluster_dist_kl, cluster_dist_l1, silhouette_diff, min_cluster_ratio

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
elif name == 'MTFL':
  dataset = MTFL_data.MTFL()
  X, y, s = dataset.data
elif name == 'Yale_alter':
  dataset = ExtendedYaleB_alter(resize=True)
  X, y, s = dataset.data
else:
  print('Invalid dataset name')
  sys.exit()

# cl_algo can only be FSC or SFD
if cl_algo != 'FSC' and cl_algo != 'SFD' and cl_algo != 'KFC':
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

def perform_combined_attack(bal_weight, ent_weight, seed, budget):
    def combined_attack(solution):
        balance_score = attack_balance(solution)
        entropy_score = attack_entropy(solution)
        return bal_weight * balance_score + ent_weight * entropy_score
    
    dim = Dimension(dim_size, [[0, 1]]*dim_size, [False]*dim_size)
    obj = Objective(combined_attack, dim)
    solution = Opt.min(obj, Parameter(budget=budget))
    balance_score = process_solution(solution)[0]
    return balance_score


def attack_min_cluster_ratio(solution):
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

  group_a = (s_eval == 0)
  group_b = (s_eval == 1)

  min_cluster_ratio_val = min_cluster_ratio(group_a, group_b, labels_sfd_eval)

  return min_cluster_ratio_val

def attack_silhouette_score(solution):
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

  sil_score = silhouette(X_eval, labels_sfd_eval) if np.unique(labels_sfd_eval).shape[0] > 1 else "N/A"

  return sil_score

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

  s_eval = np.array([s_copy[idx] for idx in V_idx])
  X_eval = np.array([X_copy[idx] for idx in V_idx])
  labels_sfd_eval = np.array([labels_sfd[idx] for idx in V_idx])
  y_eval = np.array([y_copy[idx] for idx in V_idx])
  group_a = (s_eval == 0)
  group_b = (s_eval == 1)
  

  bal = balance(labels_sfd_eval, X_eval, s_eval)
  min_cluster_ratio_val = min_cluster_ratio(group_a, group_b, labels_sfd_eval)
  cluster_dist_l1_val = cluster_dist_l1(group_a, group_b, labels_sfd_eval)
  cluster_dist_kl_val = cluster_dist_kl(group_a, group_b, labels_sfd_eval)
  sil_diff = silhouette_diff(group_a, group_b, X_eval, labels_sfd_eval) if np.unique(labels_sfd_eval).shape[0] > 1 else "N/A"
  ent = entropy(labels_sfd_eval, s_eval)
  ent_a = cluster_dist_entropy(group_a, labels_sfd_eval)
  ent_b = cluster_dist_entropy(group_b, labels_sfd_eval)
  accuracy = acc(y_eval, labels_sfd_eval)
  nmi_score = nmi(y_eval, labels_sfd_eval)
  ari_score = ari(y_eval, labels_sfd_eval)
  sil_score = silhouette(X_eval, labels_sfd_eval) if np.unique(labels_sfd_eval).shape[0] > 1 else "N/A"

  return (bal, min_cluster_ratio_val, cluster_dist_l1_val, cluster_dist_kl_val, sil_diff, ent, ent_a, ent_b, accuracy, nmi_score, ari_score, sil_score)

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

  s_eval = np.array([s_copy[idx] for idx in V_idx])
  X_eval = np.array([X_copy[idx] for idx in V_idx])
  labels_sfd_eval = np.array([labels_sfd[idx] for idx in V_idx])
  y_eval = np.array([y_copy[idx] for idx in V_idx])
  group_a = (s_eval == 0)
  group_b = (s_eval == 1)

  bal = balance(labels_sfd_eval, X_eval, s_eval)
  min_cluster_ratio_val = min_cluster_ratio(group_a, group_b, labels_sfd_eval)
  cluster_dist_l1_val = cluster_dist_l1(group_a, group_b, labels_sfd_eval)
  cluster_dist_kl_val = cluster_dist_kl(group_a, group_b, labels_sfd_eval)
  sil_diff = silhouette_diff(group_a, group_b, X_eval, labels_sfd_eval) if np.unique(labels_sfd_eval).shape[0] > 1 else "N/A"
  ent = entropy(labels_sfd_eval, s_eval)
  ent_a = cluster_dist_entropy(group_a, labels_sfd_eval)
  ent_b = cluster_dist_entropy(group_b, labels_sfd_eval)
  accuracy = acc(y_eval, labels_sfd_eval)
  nmi_score = nmi(y_eval, labels_sfd_eval)
  ari_score = ari(y_eval, labels_sfd_eval)
  sil_score = silhouette(X_eval, labels_sfd_eval) if np.unique(labels_sfd_eval).shape[0] > 1 else "N/A"

  return (bal, min_cluster_ratio_val, cluster_dist_l1_val, cluster_dist_kl_val, sil_diff, ent, ent_a, ent_b, accuracy, nmi_score, ari_score, sil_score)

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
    elif cl_algo == 'KFC': # I randomly chose 20, need to motivate for paper
      return 20
  elif name == 'Office-31':
    return 20
  elif name == 'Yale':
    if cl_algo == 'FSC':
      return 10
    elif cl_algo == 'SFD':
      return 20
    elif cl_algo == 'KFC': # I randomly chose 20, need to motivate for paper
      return 20
  elif name == 'DIGITS':
    if cl_algo == 'FSC':
      return 15
    elif cl_algo == 'SFD':
      return 25
    elif cl_algo == 'KFC': # I randomly chose 20, need to motivate for paper
      return 20
  elif name == "MTFL":
     return 50 # I tried 50 arbitrarly for now, let's wait for author's response
  elif name == "Yale_alter":
     return 20 # I tried 20 arbitrarly for now, let's wait for author's response

def select_clustering_algorithm(name, cl_algo, n_clusters, random_state):
  '''Selects the clustering algorithm based on the dataset and the clustering algorithm name'''
  metric_str = 'manhattan' if name == 'MNIST_USPS' else 'euclidean'
  if cl_algo == 'FSC':
      return FairSpectral(n_clusters=n_clusters, num_neighbors=3, metric_str=metric_str, random_state=random_state)
  elif cl_algo == 'SFD':
      beta = 1 if name == 'DIGITS' else 2
      return ScalableFairletDecomposition(n_clusters=n_clusters, alpha=5, beta=beta, random_state=random_state)
  elif cl_algo == 'KFC':
      return FairKCenter(n_clusters=n_clusters, delta=0.1, random_state=random_state)

def print_results(result_dict, result_name, name, cl_algo):
    '''Prints and saves the results in a pickle file.'''
    print("")
    print(f'{result_name} Results for {name} and {cl_algo}')
    print("=========================================")
    for metric in result_dict.keys():
        if "N/A" in result_dict[metric]:
            print(f'{metric}: {result_dict[metric]}')
            continue
        mean_val = np.mean(result_dict[metric])
        std_val = np.std(result_dict[metric])
        print(f'{metric}: Mean = {mean_val}, Std = {std_val}')

def save_attack_data(name, cl_algo, pre_attack_res, post_attack_res, random_attack_res):
    """
    Saves the attack data in nested folders.

    Parameters:
    name (str): The name of the dataset or experiment.
    cl_algo (str): The classification algorithm used.
    pre_attack_res, post_attack_res, random_attack_res: Data to be saved.
    """

    # Create the folder structure
    folder_path = os.path.join('extra_metrics_results/combined_attack_grid_search', name, cl_algo)
    os.makedirs(folder_path, exist_ok=True)

    # Save the files
    with open(os.path.join(folder_path, 'pre_attack_res.pkl'), 'wb') as f:
        pickle.dump(pre_attack_res, f)

    with open(os.path.join(folder_path, 'post_attack_res.pkl'), 'wb') as f:
        pickle.dump(post_attack_res, f)

    with open(os.path.join(folder_path, 'random_attack_res.pkl'), 'wb') as f:
        pickle.dump(random_attack_res, f)

def create_objective(name, cl_algo, dim_size, attack_balance, attack_entropy, attack_min_cluster_ratio):
    dim = Dimension(dim_size, [[0, 1]]*dim_size, [False]*dim_size)
    return Objective(attack_balance, dim) # Trying out new attack
    if name == 'Office-31':
        if cl_algo == 'KFC':
          return Objective(attack_entropy, dim)
        else:
          return Objective(attack_balance, dim)
    elif name in ['MNIST_USPS', 'DIGITS']:
        if cl_algo == 'SFD':
            return Objective(attack_balance, dim)
        elif cl_algo == 'FSC' or cl_algo == 'KFC':
            return Objective(attack_entropy, dim)
    elif name == 'Yale':
        return Objective(attack_entropy, dim)
    elif name == 'MTFL':
        return Objective(attack_balance, dim)
    elif name == "Yale_alter":
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

# Check if the indices exist
if not os.path.exists('U_idx_' + name + '.npy') or not os.path.exists('V_idx_' + name + '.npy'):
  print('The indices do not exist.') #TODO: We need to generate the indices for the dataset
  sys.exit()
U_idx_full, V_idx_full = np.load('U_idx_' + name + '.npy').tolist(), np.load('V_idx_' + name + '.npy').tolist()

# Calculate the 15% index
j = int(0.15 * len(U_idx_full))

# Initialize result dictionaries
pre_attack_res = {'BALANCE': [], 'MIN_CLUSTER_RATIO': [], 'CLUSTER_DIST_L1': [], 'CLUSTER_DIST_KL': [], 'SILHOUETTE_DIFF': [],'ENTROPY': [], 'ENTROPY_GROUP_A': [], 'ENTROPY_GROUP_B': [], 'ACC': [], 'NMI': [], 'ARI': [], 'SIL': []}
post_attack_res = {'BALANCE': [], 'MIN_CLUSTER_RATIO': [], 'CLUSTER_DIST_L1': [], 'CLUSTER_DIST_KL': [], 'SILHOUETTE_DIFF': [],'ENTROPY': [], 'ENTROPY_GROUP_A': [], 'ENTROPY_GROUP_B': [], 'ACC': [], 'NMI': [], 'ARI': [], 'SIL': []}
random_attack_res = {'BALANCE': [], 'MIN_CLUSTER_RATIO': [], 'CLUSTER_DIST_L1': [], 'CLUSTER_DIST_KL': [], 'SILHOUETTE_DIFF': [],'ENTROPY': [], 'ENTROPY_GROUP_A': [], 'ENTROPY_GROUP_B': [], 'ACC': [], 'NMI': [], 'ARI': [], 'SIL': []}

# Update the indices for the current percentage
U_idx = U_idx_full[:j]
V_idx = V_idx_full
dim_size = len(U_idx)
initial_seeds = [42, 123, 456]  # Example seeds for initial grid search
# Example ranges for weights and budget
balance_weights = [1, 5, 10]
entropy_weights = [1, 5, 10]
budgets = [20, 40, 60]

best_score = float('inf')
best_configuration = None

for bal_weight in balance_weights:
    for ent_weight in entropy_weights:
        for budget in budgets:
            print(f"Trying out configuration: {bal_weight}, {ent_weight}, {budget}")
            combined_scores = []

            for seed in initial_seeds:
                random_state = seed
                score = perform_combined_attack(bal_weight, ent_weight, seed, budget)
                combined_scores.append(score)

            avg_score = sum(combined_scores) / len(combined_scores)
            if avg_score < best_score:
                best_score = avg_score
                best_configuration = (bal_weight, ent_weight, budget)

print("Best Configuration:", best_configuration)
print("Best Average Combined Score:", best_score)