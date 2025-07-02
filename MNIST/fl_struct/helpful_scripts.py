import torch
import os
import pynvml
import random
import numpy as np
from enum import Enum

# Class to aggregate parameters of trained models
class Aggregators(object):

    @staticmethod
    # Function to aggregate parameters using federated averaging
    def fedavg_aggregate(serialized_params_list, weights=None):
        # Default to equal weights if none provided
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        # Convert weights to a tensor if they aren't already
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        # Normalize weights
        weights = weights / torch.sum(weights)

        # Aggregate parameters using weighted sum
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * weights, dim=-1)

        return serialized_parameters

    @staticmethod
    # Function to aggregate partitioned parameters using federated averaging
    def partitioned_fedavg_aggregate(serialized_params_list, weights=None):
        # Default to equal weights if none provided
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        # Convert weights to a tensor if they aren't already
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        # Normalize weights
        weights = weights / torch.sum(weights)
        weights = weights.tolist()

        # Initialize list to hold aggregated partitioned parameters
        serialized_parameters_partition = []

        # Loop over each partition and aggregate parameters using weighted sum
        for num_partition_i in range(len(serialized_params_list[0])):
            agg_partition = weights[0] * serialized_params_list[0][num_partition_i]
            for num_client_i in range(1, len(serialized_params_list)):
                agg_partition += weights[num_client_i] * serialized_params_list[num_client_i][num_partition_i]
            serialized_parameters_partition.append(agg_partition)

        return serialized_parameters_partition

# Function to set random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Class to keep track of average and total values for metrics like loss and accuracy
class AverageMeter(object):

    def __init__(self):
        self.reset()

    # Reset all internal statistics to zero
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    # Update internal statistics with a new value and its frequency
    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

# Function to evaluate a model on a test dataset
def evaluate(model, criterion, test_loader):

    # Set the model to evaluation mode
    model.eval()
    # Get the device (GPU) where the model resides
    gpu = next(model.parameters()).device

    # Initialize metrics
    loss_ = AverageMeter()
    acc_ = AverageMeter()

    # Evaluate without computing gradients (for efficiency)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu) # Move inputs to GPU
            labels = labels.to(gpu) # Move labels to GPU

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Get class predictions
            _, predicted = torch.max(outputs, 1)

            # Update metrics
            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    return loss_.sum, acc_.avg

# Function to select the GPU with the most free memory for computation
def get_best_gpu():
    # Make sure GPUs are available
    assert torch.cuda.is_available()

    # Initialize the NVIDIA Management Library (pynvml)
    pynvml.nvmlInit()

    # Get the number of available devices
    deviceCount = pynvml.nvmlDeviceGetCount()

    # Get list of CUDA devices, default to all if none specified
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys() is not None:
        cuda_devices = [
            int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        ]
    else:
        cuda_devices = range(deviceCount)

    # Ensure devices in environment variable are actually present
    assert max(cuda_devices) < deviceCount

    # Retrieve memory information for each GPU device
    deviceMemory = []
    for i in cuda_devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)

    # Select device with most free memory
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d" % (best_device_index))

# Enum for various message codes in a distributed computing environment
class MessageCode(Enum):
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    EvaluateParams = 3
    Exit = 4
    SetUp = 5
    Activation = 6

# Class for serialize and deserialize model parameters and gradients
class SerializationTool(object):
    @staticmethod
    # Function to serialize model gradients into a tensor
    def serialize_model_gradients(model: torch.nn.Module, cpu:bool=True) -> torch.Tensor:

        gradients = [param.grad.data.view(-1) for param in model.parameters()]
        m_gradients = torch.cat(gradients)
        if cpu:
            m_gradients = m_gradients.cpu()
        
        return m_gradients

    @staticmethod
    # Function to apply serialized gradients to a model
    def deserialize_model_gradients(model: torch.nn.Module, gradients: torch.Tensor) -> None:

        idx = 0
        for parameter in model.parameters():
            layer_size = parameter.grad.numel()
            shape = parameter.grad.shape

            parameter.grad.data[:] = gradients[idx:idx+layer_size].view(shape)[:]
            idx += layer_size

    # More functions to serialize and deserialize models
    @staticmethod
    # Serialize the entire model parameters into a single tensor
    def serialize_model(model: torch.nn.Module, cpu:bool=True) -> torch.Tensor:

        parameters = [param.data.view(-1) for param in model.state_dict().values()]
        m_parameters = torch.cat(parameters)
        if cpu:
            m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    # Deserialize a tensor back into model parameters
    def deserialize_model(model: torch.nn.Module,
                      serialized_parameters: torch.Tensor,
                      mode="copy"):
        
        current_index = 0  

        for param in model.state_dict().values():
            numel = param.numel()
            size = param.size()
            if mode == "copy":
                param.copy_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "add":
                param.add_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "sub":
                param.sub_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            current_index += numel


    @staticmethod
    def serialize_trainable_model(model: torch.nn.Module, cpu:bool=True) -> torch.Tensor:

        # Flattening and concatenating all trainable parameters into a single tensor
        parameters = [param.data.view(-1) for param in model.parameters()]
        m_parameters = torch.cat(parameters)
        if cpu:
            m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
    def deserialize_trainable_model(model: torch.nn.Module,
                          serialized_parameters: torch.Tensor,
                          mode="copy"):

        current_index = 0 
        for parameter in model.parameters():
            # Get the number of elements for this parameter
            numel = parameter.data.numel()
            # Get the shape of this parameter
            size = parameter.data.size()
            if mode == "copy":
                parameter.data.copy_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "add":
                parameter.data.add_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            elif mode == "sub":
                parameter.data.sub_(
                    serialized_parameters[current_index:current_index +
                                          numel].view(size))
            current_index += numel
    
# Class for random sampling with optional probabilities
class RandomSampler():
    # Constructor
    # n: The number of items to choose from
    # probs: The probabilities associated with each item; defaults to uniform if not specified
    def __init__(self, n, probs=None):
        # Assign a name for the sampling technique being used
        self.name = "random_sampling"

        # Number of items to sample from
        self.n = n

        # Probabilities for each item. If not provided, set to uniform distribution
        self.p = probs if probs is not None else np.ones(n) / float(n)

        # Ensure the probabilities array has the same length as the number of items
        assert len(self.p) == self.n

    # Function to sample 'k' items from the 'n' items
    # k: The number of items to sample
    # replace: Whether to sample with replacement or not
    def sample(self, k, replace=False):
        # If k equals n, return all items
        if k == self.n:
            self.last_sampled = np.arange(self.n), self.p
            return np.arange(self.n)
        else:
            # Randomly sample 'k' items according to the given probabilities
            sampled = np.random.choice(self.n, k, p=self.p, replace=replace)

            # Store the last sampled items and their corresponding probabilities
            self.last_sampled = sampled, self.p[sampled]

            # Return the sorted list of sampled items
            return np.sort(sampled)

    # Function to update the probabilities
    # probs: The new probabilities
    def update(self, probs):
        self.p = probs