from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms
import torch
import torchvision
from . import split_functions as F
from .all_dataset import *
from PIL import Image
from abc import ABC, abstractmethod

class Subset(Dataset):

    def __init__(self, dataset, indices,
                 transform=transforms.Compose(
                     [transforms.ToPILImage(), transforms.ToTensor()]),
                 target_transform=None):
        self.data = []
        for idx in indices:
            self.data.append(dataset.data[idx])

        if not isinstance(dataset.targets, np.ndarray):
            dataset.targets = np.array(dataset.targets)

        self.targets = dataset.targets[indices].tolist()

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        img, label = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.targets)


class FedDataset(object):
    def __init__(self) -> None:
        self.num = None  
        self.root = None 
        self.path = None  

    def preprocess(self):
        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

    def get_dataset(self, id, type="train"):
        raise NotImplementedError()

    def get_dataloader(self, id, batch_size, type="train"):
        raise NotImplementedError()

    def __len__(self):
        return self.num


class BasicPartitioner():

    num_classes = 2

    def __init__(self, targets, num_clients,
                 partition='iid',
                 dir_alpha=None,
                 major_classes_num=1,
                 verbose=True,
                 min_require_size=None,
                 seed=None):
        self.targets = np.array(targets) 
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        self.client_dict = dict()
        self.partition = partition
        self.dir_alpha = dir_alpha
        self.verbose = verbose
        self.min_require_size = min_require_size

        np.random.seed(seed)

        self.client_dict = self._perform_partition()
        self.client_sample_count = F.samples_num_count(
            self.client_dict, self.num_clients)

    def _perform_partition(self):
        
        if self.partition == "unbalance":
            client_sample_nums = F.dirichlet_unbalance_split(self.num_clients, self.num_samples,
                                                             self.dir_alpha)
            client_dict = F.homo_partition(
                client_sample_nums, self.num_samples)

        else:
            # IID
            client_sample_nums = F.balance_split(
                self.num_clients, self.num_samples)
            client_dict = F.homo_partition(
                client_sample_nums, self.num_samples)

        return client_dict

    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return len(self.client_dict)


class MNISTPartitioner(BasicPartitioner):
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

        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform
        self.targt_transform = target_transform

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

        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))

        trainset = torchvision.datasets.MNIST(root=self.root,
                                              train=True,
                                              download=download)

        partitioner = MNISTPartitioner(trainset.targets,
                                       self.num_clients,
                                       partition=partition,
                                       dir_alpha=dir_alpha,
                                       verbose=verbose,
                                       seed=seed)

        subsets = {
            cid: Subset(trainset,
                        partitioner.client_dict[cid],
                        transform=transform,
                        target_transform=target_transform)
            for cid in range(self.num_clients)
        }
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "train", "data{}.pkl".format(cid)))

    def get_dataset(self, cid, type="train"):

        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(cid)))
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train"):

        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader


class CIFARSubset(Subset):
    def __init__(self,
                 dataset,
                 indices,
                 transform=None,
                 target_transform=None,
                 to_image=True):
        self.data = []
        for idx in indices:
            if to_image:
                self.data.append(Image.fromarray(dataset.data[idx]))
        if not isinstance(dataset.targets, np.ndarray):
            dataset.targets = np.array(dataset.targets)
        self.targets = dataset.targets[indices].tolist()
        self.transform = transform
        self.target_transform = target_transform

class DataPartitioner(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def _perform_partition(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

class CIFAR10Partitioner(DataPartitioner):
    num_classes = 10

    def __init__(self, targets, num_clients,
                 balance=True, partition="iid",
                 unbalance_sgm=0,
                 num_shards=None,
                 dir_alpha=None,
                 verbose=True,
                 min_require_size=None,
                 seed=None):

        self.targets = np.array(targets)  # with shape (num_samples,)
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        self.client_dict = dict()
        self.partition = partition
        self.balance = balance
        self.dir_alpha = dir_alpha
        self.num_shards = num_shards
        self.unbalance_sgm = unbalance_sgm
        self.verbose = verbose
        self.min_require_size = min_require_size
        # self.rng = np.random.default_rng(seed)  # rng currently not supports randint
        np.random.seed(seed)

        # partition scheme check
        if balance is None:
            assert partition in ["dirichlet", "shards"], f"When balance=None, 'partition' only " \
                                                         f"accepts 'dirichlet' and 'shards'."
        elif isinstance(balance, bool):
            assert partition in ["iid", "dirichlet"], f"When balance is bool, 'partition' only " \
                                                      f"accepts 'dirichlet' and 'iid'."
        else:
            raise ValueError(f"'balance' can only be NoneType or bool, not {type(balance)}.")

        # perform partition according to setting
        self.client_dict = self._perform_partition()
        # get sample number count for each client
        self.client_sample_count = F.samples_num_count(self.client_dict, self.num_clients)

    def _perform_partition(self):
        if self.balance is None:
            if self.partition == "dirichlet":
                client_dict = F.hetero_dir_partition(self.targets,
                                                     self.num_clients,
                                                     self.num_classes,
                                                     self.dir_alpha,
                                                     min_require_size=self.min_require_size)

            else:  # partition is 'shards'
                client_dict = F.shards_partition(self.targets, self.num_clients, self.num_shards)

        else:  # if balance is True or False
            # perform sample number balance/unbalance partition over all clients
            if self.balance is True:
                client_sample_nums = F.balance_split(self.num_clients, self.num_samples)
            else:
                client_sample_nums = F.lognormal_unbalance_split(self.num_clients,
                                                                 self.num_samples,
                                                                 self.unbalance_sgm)

            # perform iid/dirichlet partition for each client
            if self.partition == "iid":
                client_dict = F.homo_partition(client_sample_nums, self.num_samples)
            else:  # for dirichlet
                client_dict = F.client_inner_dirichlet_partition(self.targets, self.num_clients,
                                                                 self.num_classes, self.dir_alpha,
                                                                 client_sample_nums, self.verbose)

        return client_dict

    def __getitem__(self, index):

        return self.client_dict[index]

    def __len__(self):

        return len(self.client_dict)


class CIFAR100Partitioner(CIFAR10Partitioner):

    num_classes = 100



class PartitionCIFAR(FedDataset):

    def __init__(self,
                 root,
                 path,
                 dataname,
                 num_clients,
                 download=True,
                 preprocess=False,
                 balance=True,
                 partition="iid",
                 unbalance_sgm=0,
                 num_shards=None,
                 dir_alpha=None,
                 verbose=True,
                 seed=None,
                 transform=None,
                 target_transform=None) -> None:
        self.dataname = dataname
        self.root = os.path.expanduser(root)
        self.path = path
        self.num_clients = num_clients
        self.transform = transform
        self.targt_transform = target_transform

        if preprocess:
            self.preprocess(balance=balance,
                            partition=partition,
                            unbalance_sgm=unbalance_sgm,
                            num_shards=num_shards,
                            dir_alpha=dir_alpha,
                            verbose=verbose,
                            seed=seed,
                            download=download)

    def preprocess(self,
                   balance=True,
                   partition="iid",
                   unbalance_sgm=0,
                   num_shards=None,
                   dir_alpha=None,
                   verbose=True,
                   seed=None,
                   download=True):
        self.download = download

        if os.path.exists(self.path) is not True:
            os.mkdir(self.path)
            os.mkdir(os.path.join(self.path, "train"))
            os.mkdir(os.path.join(self.path, "var"))
            os.mkdir(os.path.join(self.path, "test"))
        # train dataset partitioning
        if self.dataname == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=self.root,
                                                    train=True,
                                                    download=self.download)
            partitioner = CIFAR10Partitioner(trainset.targets,
                                             self.num_clients,
                                             balance=balance,
                                             partition=partition,
                                             unbalance_sgm=unbalance_sgm,
                                             num_shards=num_shards,
                                             dir_alpha=dir_alpha,
                                             verbose=verbose,
                                             seed=seed)
        elif self.dataname == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root=self.root,
                                                     train=True,
                                                     download=self.download)
            partitioner = CIFAR100Partitioner(trainset.targets,
                                              self.num_clients,
                                              balance=balance,
                                              partition=partition,
                                              unbalance_sgm=unbalance_sgm,
                                              num_shards=num_shards,
                                              dir_alpha=dir_alpha,
                                              verbose=verbose,
                                              seed=seed)
        else:
            raise ValueError(
                f"'dataname'={self.dataname} currently is not supported. Only 'cifar10', and 'cifar100' are supported."
            )

        subsets = {
            cid: CIFARSubset(trainset,
                        partitioner.client_dict[cid],
                        transform=self.transform,
                        target_transform=self.targt_transform)
            for cid in range(self.num_clients)
        }
        for cid in subsets:
            torch.save(
                subsets[cid],
                os.path.join(self.path, "train", "data{}.pkl".format(cid)))

    def get_dataset(self, cid, type="train"):
        dataset = torch.load(
            os.path.join(self.path, type, "data{}.pkl".format(cid)))
        return dataset

    def get_dataloader(self, cid, batch_size=None, type="train"):
        dataset = self.get_dataset(cid, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader