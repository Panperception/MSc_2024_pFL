# Importing necessary PyTorch modules for neural network creation
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining a Multilayer Perceptron (MLP) class which inherits from the PyTorch nn.Module class
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
        self.relu = nn.ReLU()

    # Defining the forward pass through the network
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# from torch tutorial
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Implementation of the AlexNet architecture, a pioneering deep CNN model.
# This version is modified to fit datasets with different characteristics than ImageNet.
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        # Call the __init__ method of the parent class (nn.Module)
        super(AlexNet, self).__init__()

        # Define the feature extraction layers
        self.features = nn.Sequential(

            # First convolutional layer
            # 3 input channels (assumed RGB images), 64 output channels, 3x3 kernel size with stride of 2 and padding of 1
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            # ReLU activation in place
            nn.ReLU(inplace=True),
            # Max pooling with 2x2 kernel size
            nn.MaxPool2d(kernel_size=2),

            # Second convolutional layer
            # 64 input channels, 192 output channels, 3x3 kernel size with padding of 1
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            # ReLU activation in place
            nn.ReLU(inplace=True),
            # Max pooling with 2x2 kernel size
            nn.MaxPool2d(kernel_size=2),

            # Third convolutional layer
            # 192 input channels, 384 output channels, 3x3 kernel size with padding of 1
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # ReLU activation in place
            nn.ReLU(inplace=True),

            # Fourth convolutional layer
            # 384 input channels, 256 output channels, 3x3 kernel size with padding of 1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # ReLU activation in place
            nn.ReLU(inplace=True),

            # Fifth convolutional layer
            # 256 input channels, 256 output channels, 3x3 kernel size with padding of 1
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # ReLU activation in place
            nn.ReLU(inplace=True),
            # Max pooling with 2x2 kernel size
            nn.MaxPool2d(kernel_size=2),
        )
        # Define the classifier layers
        self.classifier = nn.Sequential(
            # Note: Dropout layers are commented out. Uncomment if necessary.

            # First fully connected layer
            # The input dimensions assume the spatial size is reduced to 2x2 by the end of the feature extractor
            nn.Linear(256 * 2 * 2, 4096),
            # ReLU activation in place
            nn.ReLU(inplace=True),

            # Second fully connected layer
            nn.Linear(4096, 4096),
            # ReLU activation in place
            nn.ReLU(inplace=True),

            # Third fully connected layer - the output layer
            # Outputs `num_classes` scores, one for each class
            nn.Linear(4096, num_classes),
        )

    # Defines the forward pass of the AlexNet.
    def forward(self, x):
        # Pass the input through the feature extraction layers
        x = self.features(x)

        # Flatten the feature maps so they can be fed into the classifier
        x = x.view(x.size(0), 256 * 2 * 2)

        # Pass the flattened feature maps through the classifier to get the final class scores
        x = self.classifier(x)


        return x

