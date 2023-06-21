import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Numbers(nn.Module):
    def __init__(self):
        super(Numbers, self).__init__()
        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        # Calculate the correct input size for the first fully connected layer
        self.fc_input_size = 576  # Adjusted based on the actual output shape

        self.fc1 = nn.Linear(self.fc_input_size, 128)  # Added fully connected layer
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # A

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = out.view(out.size(0), -1)

        # print(out.shape)

        out = self.fc1(out)
        out = self.relu4(out)
        out = self.fc2(out)
        return out

def train_model():
    train_dataset = dset.MNIST(root='data/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = Numbers()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model.pth')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize input images to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if __name__ == "__main__":
    train_model()
