import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transforms=None):
        self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transforms

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(
            28, 28).astype('uint8')
        # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)


class NeuralNet(nn.Module):
    """A Neural Network with a hidden layer"""

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.layer1(x)
        output = self.sigmoid(output)
        output = self.layer2(output)
        return output


input_size = 784
hidden_size = 15
output_size = 10
num_epochs = 5

learning_rate = 0.02

model = NeuralNet(input_size, hidden_size, output_size)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

transformations = transforms.Compose([transforms.ToTensor()])

custom_mnist_from_csv = \
    CustomDatasetFromCSV('./csv/train.csv', 28, 28, transformations)

mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv,
                                                batch_size=10,
                                                shuffle=False)

total_step = len(mn_dataset_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(mn_dataset_loader):
        images = images.reshape(-1, 28*28)
        out = model(images)
        loss = lossFunction(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch + \
                                                            1, num_epochs, i+1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in mn_dataset_loader:
        images = images.reshape(-1, 28*28)
        out = model(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: {} %'.format(
    100 * correct / total))
