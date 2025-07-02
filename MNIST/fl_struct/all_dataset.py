from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms
import torch
import torchvision
from . import split_functions as F
from .all_dataset import *


class Subset(Dataset):

    # Initialize the Subset with the original dataset and the indices to keep
    def __init__(self, dataset, indices,
                 transform=transforms.Compose(
                     [transforms.ToPILImage(), transforms.ToTensor()]),
                 target_transform=None):

        # Populate the data list based on the indices passed
        self.data = []
        for idx in indices:
            self.data.append(dataset.data[idx])

        # Populate the targets list based on the indices passed
        if not isinstance(dataset.targets, np.ndarray):
            dataset.targets = np.array(dataset.targets)

        self.targets = dataset.targets[indices].tolist()

        # Optional transformations for data and targets
        self.transform = transform
        self.target_transform = target_transform

    # Retrieve an item from the Subset given an index
    def __getitem__(self, index):

        img, label = self.data[index], self.targets[index]

        # Apply the optional transformations to the image and label
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    # Get the length of the Subset
    def __len__(self):
        return len(self.targets)


# Base class to handle federated datasets
class FedDataset(object):
    def __init__(self) -> None:
        # Number of clients or partitions
        self.num = None
        # Root directory of the dataset
        self.root = None
        # Full path to the dataset
        self.path = None

    # Preprocess the dataset, make directories if they don't exist
    def preprocess(self):
        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

    # Placeholder for method to get the dataset for a given client id and type
    def get_dataset(self, id, type="train"):
        raise NotImplementedError()

    # Placeholder for method to get DataLoader object for a given client id and type
    def get_dataloader(self, id, batch_size, type="train"):
        raise NotImplementedError()

    # Return the number of clients or partitions
    def __len__(self):
        return self.num


class BasicPartitioner():
    # The number of classes in the dataset
    num_classes = 2

    def __init__(self, targets, num_clients,
                 partition='iid',
                 dir_alpha=None,
                 major_classes_num=1,
                 verbose=True,
                 min_require_size=None,
                 seed=None):
        self.targets = np.array(targets) # Labels of the dataset
        self.num_samples = self.targets.shape[0] # Total number of samples
        self.num_clients = num_clients # Number of clients to partition data among
        self.client_dict = dict() # Dictionary to store partitioned data
        self.partition = partition # Partition type, can be "iid" or "unbalance"
        self.dir_alpha = dir_alpha # Alpha parameter for Dirichlet distribution
        self.verbose = verbose # Verbose output flag
        self.min_require_size = min_require_size # Minimum required size for each partition

        np.random.seed(seed) # Set random seed if provided

        # Perform partition and populate self.client_dict
        self.client_dict = self._perform_partition()

        # Count the number of samples per client
        self.client_sample_count = F.samples_num_count(
            self.client_dict, self.num_clients)

    # Method to perform data partitioning
    def _perform_partition(self):
        
        if self.partition == "unbalance":
            client_sample_nums = F.dirichlet_unbalance_split(self.num_clients, self.num_samples,
                                                             self.dir_alpha)
            client_dict = F.homo_partition(
                client_sample_nums, self.num_samples)

        else:
            # Default to IID (independently and identically distributed) partitioning
            client_sample_nums = F.balance_split(
                self.num_clients, self.num_samples)
            client_dict = F.homo_partition(
                client_sample_nums, self.num_samples)

        return client_dict

    # Allow dictionary-style access to partitions
    def __getitem__(self, index):
        return self.client_dict[index]

    # Return the number of partitions
    def __len__(self):
        return len(self.client_dict)


class MNISTPartitioner(BasicPartitioner):
    # The number of features for MNIST dataset (28x28)
    num_features = 784


class PartitionedMNIST(FedDataset):

    def __init__(self,
                 root,
                 path,
                 num_clients,
                 download=True,
                 preprocess=False,
                 partition="iid",
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 transform=None,
                 target_transform=None) -> None:

        # Initialize root directory, path, and number of clients
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform # Data transformations
        self.targt_transform = target_transform # Target/label transformations

        # Optionally preprocess and partition the dataset
        if preprocess:
            self.preprocess(partition=partition,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            download=download,
                            transform=transform,
                            target_transform=target_transform)

    def preprocess(self,
                   partition="iid",
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True,
                   transform=None,
                   target_transform=None):

        # Whether or not to download the MNIST dataset
        self.download = download

        # Create directories if they don't exist
        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        # Load the MNIST dataset
        trainset = torchvision.datasets.MNIST(root=self.root,
                                              train=True,
                                              download=download)

        # Partition the data using MNISTPartitioner
        partitioner = MNISTPartitioner(trainset.targets,
                                       self.num_clients,
                                       partition=partition,
                                       dir_alpha=dir_alpha,
                                       verbose=verbose,
                                       seed=seed)

        # Create subsets based on the partitioning
        subsets = {
            cid: Subset(trainset,
                        partitioner.client_dict[cid],
                        transform=transform,
                        target_transform=target_transform)
            for cid in range(self.num_clients)
        }

        # Save the partitioned data
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "train", "data{}.pkl".format(cid)))

    def get_dataset(self, cid, type="train"):
        # Load a saved dataset partition by client id (cid)
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(cid)))
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train"):
        # Get a DataLoader for the specified partition and batch size
        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader
