import torch

from fl_struct.maintainer import *
from fl_struct.helpful_scripts import *
from tqdm import *

# The server handler in the Scaffold algorithm
class ScaffoldServerHandler(SyncServerHandler):

    # The downlink_package property returns the current global model parameters
    # along with the global control variate that will be sent to the clients
    @property
    def downlink_package(self):
        return [self.model_parameters, self.global_c]

    # Set up the learning rate and initialize the global control variate
    def setup_optim(self, lr):
        # Set learning rate
        self.lr = lr
        # Initialize the global control variate as zeros
        # with the same shape as the model parameters
        self.global_c = torch.zeros_like(self.model_parameters)

    # Update the model parameters and the global control variate on the server
    def global_update(self, buffer):

        # Extract the model deltas and control variate deltas from the buffer
        dys = [ele[0] for ele in buffer] # List of model deltas from each client
        dcs = [ele[1] for ele in buffer] # List of control variate deltas from each client

        # Perform Federated Averaging on the received deltas for model and control variates
        dx = Aggregators.fedavg_aggregate(dys) # Aggregate model deltas
        dc = Aggregators.fedavg_aggregate(dcs) # Aggregate control variate deltas

        # Update global model parameters using aggregated delta and learning rate
        next_model = self.model_parameters + self.lr * dx
        self.set_model(next_model) # Update the model parameters

        # Update the global control variate using Federated Averaging
        # The control variate is scaled by the ratio of the number of contributing clients to total clients
        self.global_c += 1.0 * len(dcs) / self.num_clients * dc

# The client trainer in the Scaffold algorithm
class ScaffoldSerialClientTrainer(SGDSerialClientTrainer):
    
    # Set up the optimizer and initialize the control variates for each client
    def setup_optim(self, epochs, batch_size, lr):
        # Call parent class's setup_optim
        super().setup_optim(epochs, batch_size, lr)
        # Initialize control variates for all clients
        self.cs = [None for _ in range(self.num_clients)]

    # Perform local computations for each client
    def local_process(self, payload, id_list):
        # Extract model parameters from the payload
        model_parameters = payload[0]
        # Extract the global control variate from the payload
        global_c = payload[1]

        # Iterate over each client ID in the list
        for id in tqdm(id_list):
            # Get the data loader for the given client
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            # Train the model and cache the results
            pack = self.train(id, model_parameters, global_c, data_loader)
            self.cache.append(pack)

    # Train the model for each client
    def train(self, id, model_parameters, global_c, train_loader):
        # Set the model parameters
        self.set_model(model_parameters)
        # Store the original model parameters
        frz_model = model_parameters

        # Initialize the control variate for the client if it is None
        if self.cs[id] is None:
            self.cs[id] = torch.zeros_like(model_parameters)

        # Perform training over the specified number of epochs
        for _ in range(self.epochs):
            for data, target in train_loader:
                # Check for GPU availability
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                # Forward pass and compute the loss
                output = self.model(data)
                loss = self.criterion(output, target)

                # Zero the gradients, perform backpropagation and update the gradients
                self.optimizer.zero_grad()
                loss.backward()

                # Fetch the gradients
                grad = self.model_gradients
                # Adjust the gradients with control variates and global control
                grad = grad - self.cs[id] + global_c

                # Update the model's gradients
                idx = 0
                for parameter in self._model.parameters():
                    layer_size = parameter.grad.numel()
                    shape = parameter.grad.shape
                    parameter.grad.data[:] = grad[idx:idx+layer_size].view(shape)[:]
                    idx += layer_size

                # Update the model parameters
                self.optimizer.step()

        # Compute dy and dc and update the control variate for the client
        # Calculate model parameter changes (dy)
        dy = self.model_parameters - frz_model
        # Calculate control variate changes (dc)
        dc = -1.0 / (self.epochs * len(train_loader) * self.lr) * dy - global_c
        # Update the client-specific control variate
        self.cs[id] += dc

        return [dy, dc]
