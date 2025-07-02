from typing import List
import torch
from copy import deepcopy

# Importing necessary tools and datasets
from fl_struct.helpful_scripts import *
from fl_struct.maintainer import *
from fl_struct.all_dataset import *
from tqdm import tqdm
from typing import List

# This class maintains the model and its state, enabling us to manipulate the model on different devices
class ModelMaintainer(object):
    # Constructor
    # model: The PyTorch neural network model to maintain
    # cuda: Boolean indicating whether to use CUDA-enabled GPUs
    # device: The device string (e.g., 'cuda:0' or 'cpu')
    def __init__(self, model: torch.nn.Module, cuda: bool, device: str = None) -> None:
        # Store the CUDA availability status
        self.cuda = cuda

        # Create a deep copy of the model and move it to CPU
        self._model = deepcopy(model).cpu()

        # Initially set the device to 'cpu'
        self.device = 'cpu'

    # Method to set the model's parameters
    # parameters: The tensor containing model parameters
    def set_model(self, parameters: torch.Tensor):
        # Deserialize the parameters and set them to the model
        SerializationTool.deserialize_model(self._model, parameters)

    # Property to get the model
    @property
    def model(self) -> torch.nn.Module:
        return self._model

    # Property to get the model's parameters as a serialized tensor
    @property
    def model_parameters(self) -> torch.Tensor:
        # Serialize and return the model's parameters
        return SerializationTool.serialize_model(self._model)

    # Property to get the model's gradients as a serialized tensor
    @property
    def model_gradients(self) -> torch.Tensor:
        # Serialize and return the model's gradients
        return SerializationTool.serialize_model_gradients(self._model)

    # Property to get the shapes of each of the model's parameters
    @property
    def shape_list(self) -> List[torch.Tensor]:
        # Collect and return the shapes of each parameter in the model
        shape_list = [param.shape for param in self._model.parameters()]
        return shape_list


# SerialModelMaintainer extends the functionalities of ModelMaintainer,
# allowing support for multiple clients
class SerialModelMaintainer(ModelMaintainer):
    def __init__(self, model: torch.nn.Module, num_clients: int, cuda: bool, device: str = None, personal: bool = False) -> None:
        super().__init__(model, cuda, device)

        # If the model is personalized, each client will get its copy of the parameters
        if personal:
            self.parameters = [self.model_parameters for _ in range(num_clients)]  
        else:
            self.parameters = None

    # Override the set_model method to allow setting a model for a specific client using its id
    def set_model(self, parameters: torch.Tensor = None, id: int = None):
        if id is None:
            super().set_model(parameters)
        else:
            super().set_model(self.parameters[id])


# SGDSerialClientTrainer extends SerialModelMaintainer and focuses on
# client-side model training using Stochastic Gradient Descent
class SGDSerialClientTrainer(SerialModelMaintainer):

    # Initialization of the client trainer
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, personal)

        # Cache for storing temporary data
        self.cache = []

        # Number of clients
        self.num_clients = num_clients

        # Initialize dataset (assumes a FedDataset class is used)
        self.dataset = FedDataset

    # Setup dataset to be used for training
    def setup_dataset(self, dataset):
        self.dataset = dataset

    # Setup optimizer, loss function, and other training parameters
    def setup_optim(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    # Property to get the cache for uplink transmission
    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        # Clear cache
        self.cache = []
        return package

    # Method to handle local training on the client side
    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)

            # Get the data loader for the specific client
            data_loader = self.dataset.get_dataloader(id, self.batch_size)

            # Perform training and get the updated parameters
            pack = self.train(model_parameters, data_loader)

            # Append the updated parameters to cache
            self.cache.append(pack)

    # Method to train the model
    def train(self, model_parameters, train_loader):

        # Set the model parameters before training starts
        self.set_model(model_parameters)

        # Switch model to training mode
        self._model.train()

        # Loop over epochs
        for _ in range(self.epochs):
            for data, target in train_loader:

                # If CUDA is available, move data and target to GPU
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.criterion(output, target)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Backward pass
                loss.backward()

                # Update model parameters
                self.optimizer.step()

        # Return the updated model parameters
        return [self.model_parameters]


# This class implements server-side operations for federated learning
class SyncServerHandler(ModelMaintainer):

    # Initialization of the server
    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        num_clients: int = 0,
        sample_ratio: float = 1,
        cuda: bool = False,
        device=None,
        sampler=None,
    ):
        super(SyncServerHandler, self).__init__(model, cuda, device)

        # Validate sample_ratio
        assert 0.0 <= sample_ratio <= 1.0

        # Total number of clients
        self.num_clients = num_clients
        # Proportion of clients to sample in each round
        self.sample_ratio = sample_ratio
        # Sampling mechanism
        self.sampler = sampler

        # Calculate the number of clients to be used in each round
        self.round_clients = max(
            1, int(self.sample_ratio * self.num_clients)
        )

        # Initialize caches and round trackers
        self.client_buffer_cache = []

        # Total number of global rounds
        self.global_round = global_round

        # Current round
        self.round = 0

    # Get model parameters for downlink transmission
    @property
    def downlink_package(self) -> List[torch.Tensor]:
        return [self.model_parameters]

    # Get number of clients per round
    @property
    def num_clients_per_round(self):
        return self.round_clients

    # Property to check if the server should stop based on the global round
    @property
    def if_stop(self):
        return self.round >= self.global_round

    # Method to sample clients for each round
    def sample_clients(self, num_to_sample=None):
        if self.sampler is None:
            self.sampler = RandomSampler(self.num_clients)

        # Sample clients
        num_to_sample = self.round_clients if num_to_sample is None else num_to_sample
        sampled = self.sampler.sample(self.round_clients)

        # Update the number of clients participating in this round
        self.round_clients = len(sampled)

        assert self.num_clients_per_round == len(sampled)
        return sorted(sampled)

    # Update the global model with the aggregated parameters
    def global_update(self, buffer):

        # Aggregate the parameters from the buffer
        parameters_list = [ele[0] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list)

        # Deserialize the aggregated parameters into the model
        SerializationTool.deserialize_model(self._model, serialized_parameters)

    # Load the client's models, aggregate them and update the global model
    def load(self, payload: List[torch.Tensor]) -> bool:
        assert len(payload) > 0
        self.client_buffer_cache.append(deepcopy(payload))

        # Check that the buffer size does not exceed the number of clients per round
        assert len(self.client_buffer_cache) <= self.num_clients_per_round

        # If we have enough client models, perform aggregation and update the global model
        if len(self.client_buffer_cache) == self.num_clients_per_round:

            # Update the global model
            self.global_update(self.client_buffer_cache)

            # Increment the round
            self.round += 1

            # Clear the buffer
            self.client_buffer_cache = []

            # Indicates successful update
            return True
        else:
            # Indicates waiting for more client models
            return False