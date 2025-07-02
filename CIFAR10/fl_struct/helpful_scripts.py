import torch
import os
import pynvml
import random
import numpy as np
from enum import Enum
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score

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
        # Aggregate parameters
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
        # Aggregate parameters
        serialized_parameters_partition = []
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

# Class to keep track of average and total values
class AverageMeter(object):

    def __init__(self):
        self.reset()

    # Reset all statistics
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    # Update statistics with new value and count
    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

# Function to evaluate a model on a test dataset
def evaluate(model, criterion, test_loader):

    model.eval()
    gpu = next(model.parameters()).device

    y_test = []
    predictions = []
    predict_labels = []

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)

            y_test.extend(labels.cpu().numpy())

            outputs = model(inputs)

            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            predict_labels.extend(predicted.cpu().numpy())

            loss_.update(loss.item())
            acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

    # auc_score = roc_auc_score(y_test, predictions)

    print('Accuracy: {:.4f}'.format(float(accuracy_score(y_test, predict_labels))))
    print('F1 Score: {:.4f}'.format(float(f1_score(y_test, predict_labels, average='macro'))))
    print('Recall: {:.4f}'.format(float(recall_score(y_test, predict_labels, average='macro'))))
    print('Precision: {:.4f}'.format(float(precision_score(y_test, predict_labels, average='macro'))))
    # print('Sensitivity: {:.4f}'.format(float(recall_score(y_test, predict_labels))))
    # print('Specificity: {:.4f}'.format(float(recall_score(y_test, predict_labels, pos_label=0))))
    print('Confusion Matrix: \n{}'.format(confusion_matrix(y_test, predict_labels)))
    # print('AUC Score: {:.4f}'.format(auc_score))

    return loss_.sum, acc_.avg

# Function to select the GPU with the most free memory
def get_best_gpu():
    assert torch.cuda.is_available()
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()

    # Get list of CUDA devices, default to all if none specified
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys() is not None:
        cuda_devices = [
            int(device) for device in os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        ]
    else:
        cuda_devices = range(deviceCount)

    assert max(cuda_devices) < deviceCount
    # Get memory info for each device
    deviceMemory = []
    for i in cuda_devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    # Select device with most free memory
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d" % (best_device_index))

# Enum for message codes in distributed computing context
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
    def serialize_model(model: torch.nn.Module, cpu:bool=True) -> torch.Tensor:

        parameters = [param.data.view(-1) for param in model.state_dict().values()]
        m_parameters = torch.cat(parameters)
        if cpu:
            m_parameters = m_parameters.cpu()

        return m_parameters

    @staticmethod
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
            numel = parameter.data.numel()
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
    def __init__(self, n, probs=None):
        self.name = "random_sampling"
        self.n = n
        self.p = probs if probs is not None else np.ones(n) / float(n)

        assert len(self.p) == self.n

    def sample(self, k, replace=False):
        if k == self.n:
            self.last_sampled = np.arange(self.n), self.p
            return np.arange(self.n)
        else:
            sampled = np.random.choice(self.n, k, p=self.p, replace=replace)
            self.last_sampled = sampled, self.p[sampled]
            return np.sort(sampled)

    def update(self, probs):
        self.p = probs