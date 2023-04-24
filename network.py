# CS 189 HW 6: Neural Networks

# Imports for pytorch
import numpy as np
import torch
import torchvision
from torch import nn
import matplotlib
from matplotlib import pyplot as plt
import tqdm
import pickle


""" CNNs for CIFAR-10"""

# Move tensor(s) to chosen device
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
    def __iter__(self):
        for b in self.dataloader: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dataloader)

# Creating the datasets
transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

training_data = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

# Train/Validation partition
size = len(training_data)
train_size = int(0.9 * size)
val_size = size - train_size
training_data, validation_data = torch.utils.data.random_split(training_data, [train_size, val_size])


# Model

def conv_layers(in_channels, out_channels):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class VVNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(conv_layers(3, 192), conv_layers(192, 64), nn.MaxPool2d(2))
        self.residual1 = nn.Sequential(conv_layers(64, 96), conv_layers(96, 64))
        
        self.conv2 = nn.Sequential(conv_layers(64, 128), nn.MaxPool2d(2), conv_layers(128, 384), nn.MaxPool2d(2))
        self.residual2 = nn.Sequential(conv_layers(384, 384), conv_layers(384, 384))
        
        self.conv3 = nn.Sequential(conv_layers(384, 256), nn.MaxPool2d(2), conv_layers(256, 512), nn.MaxPool2d(2))
        self.residual3 = nn.Sequential(conv_layers(512, 512), conv_layers(512, 512))

        self.features = [self.conv1, self.residual1, self.conv2, self.residual2, self.conv3, self.residual3]
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Flatten(), 
                                        nn.Linear(512, 10))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.residual1(out) + out
        out = self.conv2(out)
        out = self.residual2(out) + out
        out = self.conv3(out)
        out = self.residual3(out) + out
        out = self.classifier(out)
        return out


name = "VVNet_"
model = VVNet()

# Alternatively, use the following code to retrieve a pre-trained model
# with open("models/cifar_{}.pth".format(name + str(35)),"rb") as f:
#    model = pickle.load(f)

if torch.cuda.is_available():
  device = torch.device("cuda")
  model.to('cuda')
else:
  device = torch.device("cpu")
print("Using device", device)


start = 0
epochs = 35
batch_size = 50
learning_rate = 0.01

dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
dataloader = DeviceDataLoader(dataloader, device)
dataloader_val = DeviceDataLoader(dataloader_val, device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs, 
                                                steps_per_epoch=len(dataloader))

train_loss_epoches, val_loss_epoches = [], []
train_accuracy_epoches, val_accuracy_epoches = [], []

model.train() # Put model in training mode
for epoch in range(start, epochs):
    training_losses, validation_losses = [], []
    num_correct, num_correct_val = 0, 0
    for x, y in tqdm.tqdm_notebook(dataloader, unit="batch"):
        optimizer.zero_grad() # Remove the gradients from the previous step
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
        num_correct += torch.sum(torch.argmax(pred,dim=1) == y).item()
        scheduler.step()
    
    for x, y in tqdm.tqdm_notebook(dataloader_val, unit="batch"):
        pred = model(x)
        loss = criterion(pred, y)
        validation_losses.append(loss.item())
        num_correct_val += torch.sum(torch.argmax(pred,dim=1) == y).item()

    train_mean_loss = np.mean(training_losses)
    val_mean_loss = np.mean(validation_losses)
    train_accuracy = num_correct / train_size
    val_accuracy = num_correct_val / val_size
    print("Finished Epoch", epoch + 1, ", training loss:", train_mean_loss, ", validation loss:", val_mean_loss)
    train_loss_epoches.append(train_mean_loss)
    val_loss_epoches.append(val_mean_loss)
    print("training accuracy:", train_accuracy, ", validation accuracy:", val_accuracy)
    train_accuracy_epoches.append(train_accuracy) 
    val_accuracy_epoches.append(val_accuracy)
      
# Determine the final accuracy.
with torch.no_grad():
    model.eval() # Put model in eval mode
    num_correct, num_correct_val = 0, 0
    for x, y in dataloader:
        pred = model(x)
        num_correct += torch.sum(torch.argmax(pred,dim=1) == y).item()
    
    for x, y in dataloader_val:
        pred = model(x)
        num_correct_val += torch.sum(torch.argmax(pred,dim=1) == y).item()

    train_accuracy = num_correct / train_size
    val_accuracy = num_correct_val / val_size
    print("Train Accuracy:", train_accuracy)
    print("Validation Accuracy:", val_accuracy)
    model.train() # Put model back in train mode

# Save the model, uncomment if needed
#with open("models/cifar_{}.pth".format(name + str(epoch)),"wb") as f:
#    model.eval()
#    pickle.dump(model,f)
#    model.train()

plt.figure(1)
plt.subplot(111)
plt.title("Epoch vs. Training Losses")
ax = plt.gca()
ax.set_xlim([1, epochs+1])
plt.plot(range(1, epochs+1), train_loss_epoches, label="Training Losses")
plt.legend()
plt.show()



""" Testing the Model """

from PIL import Image
import os

class CIFAR10Test(torchvision.datasets.VisionDataset):
    
    def __init__(self, transform=None, target_transform=None):
        super(CIFAR10Test, self).__init__(None, transform=transform,
                                      target_transform=target_transform)
        assert os.path.exists("cifar10_test_data.npy"), "Make sure the test data file exists."
        self.data = [np.load("cifar10_test_data.npy", allow_pickle=False)]

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.data)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])

# Create the test dataset
testing_data = CIFAR10Test(
    transform=transform,
)

from torch.types import Device

test_dataloader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=False)
test_dataloader = DeviceDataLoader(test_dataloader, device)

# Store a numpy vector of the predictions for the test set in the variable `predictions`.
predictions = []
with torch.no_grad():
    model.eval() # Put model in eval mode
    for x in test_dataloader:
        pred = model(x)
        predictions.append(torch.argmax(pred, dim=1).cpu().numpy())


# This code below will generate kaggle_predictions.csv file, for Kaggle submission
import pandas as pd

if isinstance(predictions, np.ndarray):
    predictions = predictions.astype(int)
else:
    predictions = np.array(predictions, dtype=int)
predictions = predictions.reshape(len(testing_data),)
assert predictions.shape == (len(testing_data),), "Predictions were not the correct shape"
df = pd.DataFrame({'Category': predictions})
df.index += 1  # Ensures that the index starts at 1. 
df.to_csv('submission_{}.csv'.format(name + str(epochs)), index_label='Id')
