import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define hyperparameters
hiddenSize = 256
outputSize = 10
batchSize = 32
dropout = 0.3
learningRate = 0.0005
num_clients = 10
local_epochs = 5
num_rounds = 5


def load_dataset():
    inputSize = 28*28
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]
    )
    # Define train set
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Define test set
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size = batchSize, shuffle=False)

    return inputSize, trainloader, testloader


# Initialize the weights of each layer
def initialize_weights(m):
    # Check to make sure weight initialization is done only on linear layers
    if isinstance(m, nn.Linear):
        # Initialize parameters using He initialization
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ============================================= HW4====================================================

# Split function to give the entire dataset to all 10 clients
def split1(trainset):
    # Dictionary to hold indices for each class (digits)
    class_split = {}
    # For each digit from 0-9, initialize an empty list
    for number in range(10):
        class_split[number] = []
    # Add indices to dictionary
    for index, sample in enumerate(trainset):
        _, label = sample
        class_split[label].append(index)
    # Shuffle indices for randomization
    for i in class_split.values():
        np.random.shuffle(i)

    # Now we need to give each client an equal amount of data from each digit class
    client_split = {j: [] for j in range(num_clients)}
    for digit, indices in class_split.items():
        split_size = len(indices) // num_clients

        for client_id in range(num_clients):
            start_idx = client_id * split_size
            end_idx = start_idx + split_size
            client_split[client_id].extend(indices[start_idx:end_idx])
    
    # Create a list of loaded datasets for clients
    client_loaders = []
    for client_id in range(num_clients):
        client_dataset = Subset(trainset, client_split[client_id])
        client_loader = DataLoader(client_dataset, batch_size=batchSize, shuffle = True)
        client_loaders.append(client_loader)
    
    # Return the list of loaded clients
    return client_loaders


def split2(trainset):
    # Dictionary to hold indices for each class (digits)
    class_split = {}
    # For each digit from 0-9, initialize an empty list
    for number in range(10):
        class_split[number] = []
    # Add indices to dictionary
    for index, sample in enumerate(trainset):
        _, label = sample
        class_split[label].append(index)

    client_split = {j: [] for j in range(num_clients)}
    # Define pairs of digits to assign to clients
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    for client_id in range(5):
        digit1, digit2 = pairs[client_id]
        digit1_samples = class_split[digit1]
        digit2_samples = class_split[digit2]

        # Define how dataset samples are split
        half1_digit1 = digit1_samples[:len(digit1_samples)//2]
        half2_digit1 = digit1_samples[len(digit1_samples)//2:]
        half1_digit2 = digit2_samples[:len(digit2_samples)//2]
        half2_digit2 = digit2_samples[len(digit2_samples)//2:]

        # Assign slices of samples to clients
        client_split[client_id].extend(half1_digit1)
        client_split[client_id].extend(half1_digit2)

        # For clients 6-10, give the second half of digits
        client_split[client_id + 5].extend(half2_digit1)
        client_split[client_id + 5].extend(half2_digit2)
    
    # Create a list of loaded datasets for clients
    client_loaders = []
    for client_id in range(num_clients):
        client_dataset = Subset(trainset, client_split[client_id])
        client_loader = DataLoader(client_dataset, batch_size=batchSize, shuffle = True)
        client_loaders.append(client_loader)
    
    # Return the list of loaded clients
    return client_loaders


def split3(trainset):
    # Dictionary to hold indices for each class (digits)
    class_split = {}
    # For each digit from 0-9, initialize an empty list
    for number in range(10):
        class_split[number] = []
    # Add indices to dictionary
    for index, sample in enumerate(trainset):
        _, label = sample
        class_split[label].append(index)

    # Now we need to give each client an equal amount of data from each digit class
    client_split = {j: [] for j in range(num_clients)}

    # Assign one digit to each client
    for client_id in range(10):
        client_split[client_id].extend(class_split[client_id])
    
    # Create a list of loaded datasets for clients
    client_loaders = []
    for client_id in range(num_clients):
        client_dataset = Subset(trainset, client_split[client_id])
        client_loader = DataLoader(client_dataset, batch_size=batchSize, shuffle = True)
        client_loaders.append(client_loader)
    
    # Return the list of loaded clients
    return client_loaders


def fed_avg(model, client_models):
    global_state = model.state_dict()
    # Initialize global model paramters
    for key in global_state:
        global_state[key] = torch.zeros_like(global_state[key])

    # Sum all parameters from each client
    for client_model in client_models:
        client_state = client_model.state_dict()
        for key in global_state:
            global_state[key] += client_state[key]

    # Average the parameters
    for key in global_state:
        global_state[key] = global_state[key] / len(client_models)

    # Update the global with averaged parameters
    model.load_state_dict(global_state)


def train_local(global_model, client_data, criterion, device):
    # Retrieve the global model into local client
    client_model = global_model.to(device)

    # Specify an optimizer for local training
    optimizer = optim.Adam(client_model.parameters(), lr=learningRate)

    client_model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    count = 0
    # Training settings
    for _ in range(local_epochs):
        for inputs, labels in client_data:
            inputs = inputs.view(-1, inputSize).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = client_model(inputs)
            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Count to track number of epochs
        count += 1
        # Calculate and report average running loss at each epoch
        avg_loss = running_loss / len(client_data)
        # Calculate accuracy at each epoch
        accuracy = 100 * correct / total
        print(f"Client Training - Epoch {count}/{local_epochs} Avg Running Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        running_loss = 0.0

    return client_model


def federated_train(model, client_loaders, criterion, device, num_rounds):
    # Define list of accuracies used for graph
    accuracy_list = []
    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")
        
        client_models = []
        for client_loader in client_loaders:
            # Train each client on local data
            client_model = train_local(model, client_loader, criterion, device)
            client_models.append(client_model)

        # Apply federated averaging
        fed_avg(model, client_models)
        # Test the global model right after aggregation, return the round accuracy
        accuracy = test(model,testloader, device)
        # Add the accuracy to the list
        accuracy_list.append(accuracy)
    
    # Plot accuracy
    plt.figure(figsize=(10,6))
    plt.plot(range(1, num_rounds + 1), accuracy_list, label="Accuracy")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()


# Testing neural network
def test(model, testloader, device):
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
    print(f'Global model accuracy on test set: {accuracy:.2f}%')
    print(f'Global model F1 score on test set: {f1:.2f}%')

    # Return accuracy so it can be graphed from federated_train
    return accuracy


# Load dataset
inputSize, trainloader, testloader = load_dataset()


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


# Apply the weight initialization to the model
model.apply(initialize_weights)
# Set optimizer to Adam
optimizer = optim.Adam(model.parameters(), lr=learningRate)
# Set loss function to Cross Entropy
criterion = nn.CrossEntropyLoss()
# Cross Entropy Loss automatically uses Softmax activation, so no need to excplicitly apply Softmax

# Define the different splits for user
print("Split1: every client trains on all 10 digits")
print("Split2: every client trains on 2 of 10 digits")
print("Split3: every client trains on 1 of 10 digits")

# Ask user to select split
split = input("Please choose how to split the training data (1, 2, or 3): ")
if (split == "1"):
    client_loaders = split1(trainloader.dataset)
elif (split == "2"):
    client_loaders = split2(trainloader.dataset)
elif (split == "3"):
    client_loaders = split3(trainloader.dataset)
else:
    exit("Only inputs of 1, 2 or 3 are allowed.")

# Train using federated learning (test after each federated round)
federated_train(model, client_loaders, criterion, device, num_rounds)
