import numpy as np
from pyml import tensor
import pyml.nn
from pyml.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

class SimpleNN(pyml.nn.Module):
    def __init__(self):
        self.fc1 = pyml.nn.Linear(28 * 28, 128)
        self.fc2 = pyml.nn.Linear(128, 64)
        self.fc3 = pyml.nn.Linear(64, 10)
        self.relu = pyml.nn.Relu()
        self.softmax = pyml.nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

for inputs, labels in testloader:
    print(inputs.shape)
    print(labels.shape)

model = SimpleNN()
criterion = pyml.nn.CrossEntropyLoss()
optimizer = pyml.optim.SGD(model.parameters(), lr=0.001)

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
