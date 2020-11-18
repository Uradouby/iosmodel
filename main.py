import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def before_train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


def saveModel(net):
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


def trainModel(net, train_loader, criterion, optimizer):
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('epoch %d, %5d inputs\' average loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')


def predictModel(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def transModel_to_onnx(net):
    # 转化为onnx格式
    dummy_input = torch.rand(1, 3, 32, 32)
    input_names = ['images']
    output_names = ['classLabelProbs']
    onnx_name = 'cifar10_net.onnx'
    torch.onnx.export(net,
                      dummy_input,
                      onnx_name,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential (

            # First convolutional layer
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolutional layer
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third convolutional layer
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential (

            # Dropout layer
            nn.Dropout(p=0.1),

            # First fully connected layer
            nn.Linear(in_features=24 * 4 * 4, out_features=192),
            nn.ReLU(inplace=True),

            # Second fully connected layer
            nn.Linear(in_features=192, out_features=96),
            nn.ReLU(inplace=True),

            # Third fully connected layer
            nn.Linear(in_features=96, out_features=10),
        )

    def forward(self, x):

        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten
        x = x.view(-1, 24 * 4 * 4)

        # Fully connected layers
        x = self.fc_layers(x)

        return x


if __name__ == "__main__":
    trainloader, testloader = before_train()
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainModel(net, trainloader, criterion, optimizer)

    predictModel(net, testloader)

    saveModel(net)

    transModel_to_onnx(net)

    #print("!")

