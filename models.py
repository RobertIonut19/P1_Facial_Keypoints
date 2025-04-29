import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet1(nn.Module):
    """
    Two layers, convolutional &  AveragePooling

    Result:
        - no learning happened over the training process
    """
    def __init__(self):
        super(SimpleNet1, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        self.pool = nn.AdaptiveAvgPool2d((6, 6)) 
        
        self.fc1 = nn.Linear(32 * 6 * 6, 136)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class SimpleNet2(nn.Module):
    """
    Two layers, convolutional &  MaxPooling

    Results:
        - loss of around 0.19
    """
    def __init__(self):
        super(SimpleNet2, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)  
        
        self.pool = nn.MaxPool2d(2, 2)   

        self.fc1 = nn.Linear(32 * 46 * 46, 136) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)  # (96-5)+1 = 92 -> (32, 92, 92)
        self.pool1 = nn.MaxPool2d(2, 2)   # (32, 46, 46)
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(32, 64, 3)  # (46-3)+1 = 44
        self.pool2 = nn.MaxPool2d(2, 2)    # (64, 22, 22)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(64, 128, 2) # (22-2)+1 = 21
        self.pool3 = nn.MaxPool2d(2, 2)    # (128, 10, 10)
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 256, 1) # (10-1)+1 = 10
        # În loc să facem încă un maxpool care poate crea forme greșite, forțăm fix la 5x5
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))  # (256, 5, 5)
        self.drop4 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(256 * 5 * 5, 1000)
        self.drop5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(1000, 136)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = F.elu(self.conv4(x))
        x = self.adaptive_pool(x)
        x = self.drop4(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = F.elu(self.fc1(x))
        x = self.drop5(x)

        x = F.elu(self.fc2(x))
        x = self.drop6(x)

        x = self.fc3(x)  # output 136 coordonate
        return x
    