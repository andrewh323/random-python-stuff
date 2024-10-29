import os
import numpy
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define hyperparameters
hiddenSize = 512
outputSize = 10
batchSize = 32
dropout = 0.3
epochs = 50
learningRate = 0.0004


def load_dataset(datasetName):
    if datasetName == "CIFAR10":
        # CIFAR10 images are 3x32x32 size
        inputSize = 3*32*32
        # Normalize Dataset
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        # Define train set
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        # Define test set
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    elif datasetName == "MNIST":
        inputSize = 28*28
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))]
        )
        # Define train set
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        # Define test set
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    else:
        exit("The dataset entered is not one of CIFAR10 or MNIST. Exiting...")

    trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True)
    testloader = DataLoader(testset, batch_size = batchSize, shuffle=False)

    return inputSize, trainloader, testloader


datasetName = input("Please enter which dataset to use (CIFAR10 or MNIST): ")
inputSize, trainloader, testloader = load_dataset(datasetName)

# Building the model
model = nn.Sequential(
    # Connect input layer to hidden layer
    nn.Linear(inputSize, hiddenSize),
    # Apply ReLU activation function to hidden layer
    nn.ReLU(),
    # Set dropout
    nn.Dropout(dropout),
    # Connect hidden layer to output layer
    nn.Linear(hiddenSize, outputSize)
    # nn.Softmax(dim=1)
).to(device)


# Initialize the weights of each layer
def initialize_weights(m):
    # Check to make sure weight initialization is done only on linear layers
    if isinstance(m, nn.Linear):
        # Initialize parameters using He initialization
        nn.init.kaiming_normal(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# Apply the weight initialization to the model
model.apply(initialize_weights)
# Set optimizer to Adam
optimizer = optim.Adam(model.parameters(), lr=learningRate)
# Set loss function to Cross Entropy
criterion = nn.CrossEntropyLoss()
# Cross Entropy Loss automatically uses Softmax activation, so no need to excplicitly apply Softmax


# Training neural network
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            # Flatten dataset to 2 dimensions
            inputs = inputs.view(-1, inputSize).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Calculate loss function
            loss = criterion(outputs, labels)
            # Back propagation
            loss.backward()
            # Update weights
            optimizer.step()
            running_loss += loss.item()
            # Report progress every 300 steps
            if i % 300 == 299:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

# Testing neural network
def test(model, testloader, device, datasetName):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in testloader:
            # Flatten dataset to 2 dimensions
            inputs = inputs.view(-1, inputSize).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    # Calculate accuracy
    accuracy = 100*correct/total
    # Calculate F1 Score
    f1 = 100*f1_score(all_labels, all_predictions, average='weighted')
    print(f'Accuracy on {datasetName} test set: {accuracy:.2f}%')
    print(f'F1 Score on {datasetName} test set: {f1:.2f}%')
    
train(model, trainloader, criterion, optimizer, device)
test(model, testloader, device, datasetName)
