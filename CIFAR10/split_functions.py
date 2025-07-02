# Importing necessary libraries for array and data frame operations
import numpy as np
import pandas as pd

# Function to split indices of samples among clients
def split_indices(num_cumsum, rand_perm):
    # Creating a list of tuples where each tuple contains a client id and their corresponding sample indices
    client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    # Convert the list of tuples into a dictionary
    client_dict = dict(client_indices_pairs)
    return client_dict

# Function to evenly split samples among clients
def balance_split(num_clients, num_samples):
    # Calculate the number of samples each client should get
    num_samples_per_client = int(num_samples / num_clients)
    # Create an array of size 'num_clients' with each element being the number of samples per client
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(
        int)
    return client_sample_nums

# Function to unevenly split samples among clients using a Dirichlet distribution
def dirichlet_unbalance_split(num_clients, num_samples, alpha):
    min_size = 0
    # Ensure each client gets at least 10 samples
    while min_size < 10:
        # Generate proportions using a Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * num_samples)

    # Calculate the number of samples for each client based on the generated proportions
    client_sample_nums = (proportions * num_samples).astype(int)
    return client_sample_nums

# Function to partition samples among clients
def homo_partition(client_sample_nums, num_samples):
    # Create an array of indices and shuffle it
    rand_perm = np.random.permutation(num_samples)
    # Calculate the cumulative sum of the number of samples for each client
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    # Split the indices among clients
    client_dict = split_indices(num_cumsum, rand_perm)
    return client_dict

# Function to count the number of samples for each client
def samples_num_count(client_dict, num_clients):
    # Create a list of tuples where each tuple contains a client id and the number of samples they have
    client_samples_nums = [[cid, client_dict[cid].shape[0]] for cid in
                           range(num_clients)]
    # Convert the list of tuples into a data frame
    client_sample_count = pd.DataFrame(data=client_samples_nums,
                                       columns=['client', 'num_samples']).set_index('client')
    return client_sample_count