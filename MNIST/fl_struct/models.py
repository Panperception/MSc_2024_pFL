# Importing necessary PyTorch modules for neural network creation
import torch.nn as nn

# Defining a Multilayer Perceptron (MLP) class which inherits from the PyTorch nn.Module class
class MLP(nn.Module):

    # Initialization function where the layers are defined
    def __init__(self, input_size, output_size):
        # Call to the parent class (nn.Module) constructor
        super(MLP, self).__init__()

        # First fully connected layer with input size and 200 hidden neurons
        self.fc1 = nn.Linear(input_size, 200)

        # Second fully connected layer with 200 hidden neurons
        self.fc2 = nn.Linear(200, 200)

        # Third fully connected layer with output size neurons
        self.fc3 = nn.Linear(200, output_size)

        # ReLU activation function
        self.relu = nn.ReLU()

    # Defining the forward pass through the network
    def forward(self, x):
        # Flattening the input tensor (used for handling multidimensional input)
        x = x.view(x.shape[0], -1)

        # First layer followed by ReLU activation
        x = self.relu(self.fc1(x))

        # Second layer followed by ReLU activation
        x = self.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)
        return x


# A Convolutional Neural Network (CNN) class designed to work with the MNIST dataset.
# The MNIST dataset consists of handwritten digit images and this CNN is structured to classify them.
class CNN(nn.Module):

    def __init__(self):
        # Call the __init__ method of the parent class (nn.Module)
        # Note: There's a typo in the code; it should be `super(CNN, self)`
        super(CNN_MNIST, self).__init__()

        # Define the first convolutional layer
        # 1 input channel (because MNIST is grayscale), 32 output channels, and a kernel size of 5x5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5))

        # Define the second convolutional layer
        # 32 input channels (from the output of the first conv layer), 64 output channels, and a kernel size of 5x5
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 5))

        # Define a max-pooling layer with a kernel size of 2x2
        # This will reduce the spatial dimensions of our feature maps by half
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Define the first fully connected (FC) layer
        # We assume the input will be flattened to 1024 dimensions and the output is 512 dimensions
        self.fc1 = nn.Linear(in_features=1024, out_features=512)

        # Activation function (ReLU) used after the first FC layer
        self.relu = nn.ReLU()

        # Define the second FC layer, which reduces the dimensions from 512 to 10
        # The output dimension of 10 is because MNIST has 10 classes (digits 0-9)
        self.fc2 = nn.Linear(512, 10)

    # Defines the forward pass of the CNN
    def forward(self, x):
        # Pass the input through the first convolutional layer followed by the max-pooling layer
        x = self.pool(self.conv1(x))

        # Pass the result through the second convolutional layer followed by another max-pooling layer
        x = self.pool(self.conv2(x))

        # Flatten the feature maps so they can be fed into the FC layer
        # x.shape[0] is the batch size
        x = x.view(x.shape[0], -1)

        # Pass the flattened feature maps through the first FC layer followed by a ReLU activation
        x = self.relu(self.fc1(x))

        # Pass the result through the second FC layer
        x = self.fc2(x)

        # Return the final output
        return x