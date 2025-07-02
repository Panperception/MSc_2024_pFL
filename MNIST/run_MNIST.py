# Import necessary libraries and modules
from munch import Munch # To create a simple dictionary-like object

from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fl_struct.helpful_scripts import * # Import utility functions
from fl_struct.models import * # Import pre-defined models
from fl_struct.maintainer import * # Import utilities for data and model management
from fl_struct.all_dataset import * # Import dataset-related utilities

# Import different Federated Learning algorithms
from fl_struct.algorithm.fedprox import FedProxServerHandler, FedProxSerialClientTrainer
from fl_struct.algorithm.scaffold import ScaffoldSerialClientTrainer, ScaffoldServerHandler
from fl_struct.algorithm.ditto import DittoServerHandler, DittoSerialClientTrainer
from fl_struct.algorithm.myscheme import CustomSerialClientTrainer, CustomServerHandler

# Create a configuration object using Munch
args = Munch()
args.total_client = 10 # Number of clients
args.com_round = 100 # Number of communication rounds
args.sample_ratio = 0.2 # The ratio of clients sampled per round
args.batch_size = 600 # Batch size for training
args.epochs = 5 # Number of epochs
args.lr = 0.1 # Learning rate
args.preprocess = False # Preprocessing flag
args.seed = 0 # Random seed for reproducibility

# Algorithm to be used ("fedavg", "fedprox", "scaffold", "ditto", "myscheme")
args.alg = "myscheme"

args.mu = 0.1 # Mu parameter for FedProx algorithm
args.alpha = 0.01 # Alpha parameter (could be for Dirichlet distribution in data partitioning)

# Set seed for reproducibility
setup_seed(args.seed)

# Load the test data from MNIST dataset
test_data = torchvision.datasets.MNIST(root="./datasets/mnist/",
                                       train=False,
                                       transform=transforms.ToTensor(), download=True)

# Create a dataloader for the test data
test_loader = DataLoader(test_data, batch_size=1024)

# Initialize a model
model = MLP(784, 10)

# Conditional block to set up the server handler and client trainer based on chosen algorithm
# Server Handler is responsible for aggregating client models
# Client Trainer is responsible for training models on each client
if args.alg == "fedavg":
    handler = SyncServerHandler(model=model,
                                global_round=args.com_round,
                                sample_ratio=args.sample_ratio, num_clients=args.total_client)
    trainer = SGDSerialClientTrainer(model, args.total_client, cuda=False)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)
    
if args.alg == "fedprox":
    handler = FedProxServerHandler(model=model,
                                   global_round=args.com_round,
                                   sample_ratio=args.sample_ratio, num_clients=args.total_client)
    trainer = FedProxSerialClientTrainer(model, args.total_client, cuda=False)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr, mu=args.mu)

if args.alg == "scaffold":
    handler = ScaffoldServerHandler(model=model,
                                    global_round=args.com_round,
                                    sample_ratio=args.sample_ratio, num_clients=args.total_client)
    handler.setup_optim(lr=args.lr)

    trainer = ScaffoldSerialClientTrainer(model, args.total_client, cuda=False)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

if args.alg == "ditto":
    handler = DittoServerHandler(model=model,
                                 global_round=args.com_round,
                                 sample_ratio=args.sample_ratio, num_clients=args.total_client)
    trainer = DittoSerialClientTrainer(model, args.total_client, cuda=False)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)
    
if args.alg == "myscheme":
    handler = CustomServerHandler(model=model,
                                 global_round=args.com_round,
                                 sample_ratio=args.sample_ratio, num_clients=args.total_client)
    trainer = CustomSerialClientTrainer(model, args.total_client, cuda=False)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

# Load and partition the MNIST dataset among clients
mnist = PartitionedMNIST(root='./datasets/mnist/', path="./datasets/mnist/pathmnist",
                         num_clients=args.total_client, partition='unbalance', dir_alpha=0.05, preprocess=True, transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]))

# Set up the dataset for the trainer
trainer.setup_dataset(mnist)

# Initialize round counter and accuracy list
round = 1
accuracy = []

# Set the number of clients for the handler
handler.num_clients = trainer.num_clients

# Start the training process
while handler.if_stop is False:
    # Sample clients
    sampled_clients = handler.sample_clients()
    # Get the broadcast from handler
    broadcast = handler.downlink_package

    # Start local process
    trainer.local_process(broadcast, sampled_clients)
    # Get the upload package from trainer
    uploads = trainer.uplink_package

    # Load the upload package to handler
    for pack in uploads:
        handler.load(pack)

    # Evaluate the model
    loss, acc = evaluate(trainer._model, nn.CrossEntropyLoss(), test_loader)
    # Append the accuracy to the list
    accuracy.append(acc)
    # Print the current round and test accuracy
    print("Round {}, Test Acc: {:.4f}".format(round, acc))
    # Increment the round counter
    round += 1
