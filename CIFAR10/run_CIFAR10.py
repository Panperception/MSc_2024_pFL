# Import necessary libraries and modules
from munch import Munch

from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fl_struct.helpful_scripts import *
from fl_struct.models import *
from fl_struct.maintainer import *
from fl_struct.all_dataset import *
from fl_struct.algorithm.fedprox import FedProxServerHandler, FedProxSerialClientTrainer
from fl_struct.algorithm.scaffold import ScaffoldSerialClientTrainer, ScaffoldServerHandler
from fl_struct.algorithm.ditto import DittoServerHandler, DittoSerialClientTrainer
from fl_struct.algorithm.myscheme import CustomSerialClientTrainer, CustomServerHandler

# Setup configuration parameters
args = Munch()
args.total_client = 10
args.com_round = 100
args.sample_ratio = 0.2
args.batch_size = 600
args.epochs = 5
args.lr = 0.1
args.preprocess = False
args.seed = 0
args.alg = "myscheme" 
args.mu = 0.1 
args.alpha = 0.01

# Set seed for reproducibility
setup_seed(args.seed)

# Load test data
test_data = torchvision.datasets.CIFAR10(root="./datasets/cifar-10/", 
                                         train=False, 
                                         transform=transforms.ToTensor(), 
                                         download=True)

# Create a dataloader for the test data
test_loader = DataLoader(test_data, batch_size=1024)

# Initialize a model
model = CNN_CIFAR10()
# Set up handlers and trainers based on the chosen algorithm
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

# Load the training data
# mnist = PartitionedMNIST(root='./datasets/mnist/', path="./datasets/mnist/pathmnist",
                        #  num_clients=args.total_client, partition='unbalance', dir_alpha=0.05, preprocess=True, transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()]))

cifar10 = PartitionCIFAR(root='./datasets/cifar-10/', path="./datasets/cifar-10/pathcifar10", dataname='cifar10',
                            num_clients=args.total_client, partition='iid', dir_alpha=0.05, preprocess=False, transform=transforms.Compose([transforms.ToTensor()]))

# Set up the dataset for the trainer
trainer.setup_dataset(cifar10)

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
    # # Get the upload package from trainer
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