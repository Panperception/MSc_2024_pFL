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
    def __init__(self, model: torch.nn.Module, cuda: bool, device: str = None) -> None:
        self.cuda = cuda
        # self._model = deepcopy(model).cpu()
        self._model = deepcopy(model)
        self.device = 'cuda:1'

    # Set model parameters
    def set_model(self, parameters: torch.Tensor):
        SerializationTool.deserialize_model(self._model, parameters)

    # Access model
    @property
    def model(self) -> torch.nn.Module:
        return self._model

    # Get model parameters
    @property
    def model_parameters(self) -> torch.Tensor:
        return SerializationTool.serialize_model(self._model)

    # Get model gradients
    @property
    def model_gradients(self) -> torch.Tensor:
        return SerializationTool.serialize_model_gradients(self._model)

    # Get shapes of model parameters
    @property
    def shape_list(self) -> List[torch.Tensor]:
        shape_list = [param.shape for param in self._model.parameters()]
        return shape_list


# This class extends ModelMaintainer, supporting serialization for multiple clients
class SerialModelMaintainer(ModelMaintainer):
    def __init__(self, model: torch.nn.Module, num_clients: int, cuda: bool, device: str = None, personal: bool = False) -> None:
        super().__init__(model, cuda, device)
        if personal:
            self.parameters = [self.model_parameters for _ in range(num_clients)]  
        else:
            self.parameters = None

    # Override set_model method to set model for a specific client
    def set_model(self, parameters: torch.Tensor = None, id: int = None):
        if id is None:
            super().set_model(parameters)
        else:
            super().set_model(self.parameters[id])


# This class implements client-side model training using SGD
class SGDSerialClientTrainer(SerialModelMaintainer):
    # Initialization
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self.cache = []
        self.num_clients = num_clients
        self.dataset = FedDataset

    # Setup dataset
    def setup_dataset(self, dataset):
        self.dataset = dataset

    # Setup optimizer
    def setup_optim(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    # Get package for uplink transmission
    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    # Process data locally on the client side
    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)

    # Train model
    def train(self, model_parameters, train_loader):
        self.set_model(model_parameters)
        self._model.train()

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return [self.model_parameters]


# This class implements server-side operations for federated learning
class SyncServerHandler(ModelMaintainer):
    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        num_clients: int = 0,
        sample_ratio: float = 1,
        cuda: bool = True,
        device=None,
        sampler=None,
    ):
        super(SyncServerHandler, self).__init__(model, cuda, device)

        assert 0.0 <= sample_ratio <= 1.0

        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.sampler = sampler

        # Calculate the number of clients to be used in each round
        self.round_clients = max(
            1, int(self.sample_ratio * self.num_clients)
        ) 
        self.client_buffer_cache = []

        self.global_round = global_round
        self.round = 0

    # Get model parameters for downlink transmission
    @property
    def downlink_package(self) -> List[torch.Tensor]:
        return [self.model_parameters]

    # Get number of clients per round
    @property
    def num_clients_per_round(self):
        return self.round_clients

    # Check if the server should stop
    @property
    def if_stop(self):
        return self.round >= self.global_round

    # Sample clients
    def sample_clients(self, num_to_sample=None):
        if self.sampler is None:
            self.sampler = RandomSampler(self.num_clients)

        num_to_sample = self.round_clients if num_to_sample is None else num_to_sample
        sampled = self.sampler.sample(self.round_clients)
        self.round_clients = len(sampled)

        assert self.num_clients_per_round == len(sampled)
        return sorted(sampled)

    # Update the global model with the aggregated parameters
    def global_update(self, buffer):
        parameters_list = [ele[0] for ele in buffer]
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list)
        SerializationTool.deserialize_model(self._model, serialized_parameters)

    # Load the client's models, aggregate them and update the global model
    def load(self, payload: List[torch.Tensor]) -> bool:
        assert len(payload) > 0
        self.client_buffer_cache.append(deepcopy(payload))

        assert len(self.client_buffer_cache) <= self.num_clients_per_round

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1

            self.client_buffer_cache = []

            return True 
        else:
            return False