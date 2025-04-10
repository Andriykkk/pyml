import numpy as np
from pyml import tensor

a = tensor.zeros((2, 3))
b = tensor.ones((2, 3))
c = a + b
print(c)

#  import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader

# # Define the neural network model
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 128)  # First hidden layer
#         self.fc2 = nn.Linear(128, 64)       # Second hidden layer
#         self.fc3 = nn.Linear(64, 10)        # Output layer (10 classes)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)  # Flatten the input image
#         x = self.relu(self.fc1(x))  # Apply first layer and ReLU activation
#         x = self.relu(self.fc2(x))  # Apply second layer and ReLU activation
#         x = self.fc3(x)            # Output layer
#         return x

# # Load the Fashion MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
# testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
# testloader = DataLoader(testset, batch_size=64, shuffle=False)

# # Initialize the model, loss function, and optimizer
# model = SimpleNN()
# criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# epochs = 5
# for epoch in range(epochs):
#     model.train()  # Set the model to training mode
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for inputs, labels in trainloader:
#         optimizer.zero_grad()   # Zero the gradients
#         outputs = model(inputs)  # Forward pass
#         loss = criterion(outputs, labels)  # Calculate the loss
#         loss.backward()  # Backpropagate the gradients
#         optimizer.step()  # Update weights
        
#         running_loss += loss.item()
        
#         _, predicted = torch.max(outputs, 1)  # Get predicted labels
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
    
#     print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader)}, Accuracy: {100 * correct/total}%")

# # Testing loop
# model.eval()  # Set the model to evaluation mode
# correct = 0
# total = 0
# with torch.no_grad():
#     for inputs, labels in testloader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {100 * correct/total}%")
