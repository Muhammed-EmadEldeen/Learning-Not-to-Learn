import torch
import torch.nn as nn

class Convy_base(nn.Module):
    def __init__(self, input_channels=3, dim=28, num_classes=10):
        super(Convy_base, self).__init__()
        
        self.conv1f = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batchNorm1f = nn.BatchNorm2d(16)
        self.conv2f = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchNorm2f = nn.BatchNorm2d(32)
        
        
        self.conv1g = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batchNorm1g = nn.BatchNorm2d(64)
        self.conv2g = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.batchNorm2g = nn.BatchNorm2d(64)
        self.fc_g = nn.Linear(64, num_classes)  


        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.soft = torch.nn.Softmax(dim = 1)
        

    
    def forward(self, img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))

        g = self.relu(self.conv1g(f))
        g = self.pool(self.relu(self.conv2g(g)))
        g = g.view(g.size(0), -1)
        g = self.fc_g(g)
        

        return g

    def predict_number(self, img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))

        g = self.relu(self.conv1g(f))
        g = self.pool(self.relu(self.conv2g(g)))
        g = g.view(g.size(0), -1)
        g = self.fc_g(g)
        

        return g
        
