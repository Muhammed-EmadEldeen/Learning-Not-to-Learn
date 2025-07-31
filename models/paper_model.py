import torch
import torch.nn as nn


class Convy(nn.Module):

    def __init__(self, input_channels=3, dim=28, num_classes=10):
        super(Convy, self).__init__()
        
        self.conv1f = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batchNorm1f = nn.BatchNorm2d(16)
        self.conv2f = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchNorm2f = nn.BatchNorm2d(32)
        
        
        self.conv1g = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batchNorm1g = nn.BatchNorm2d(64)
        self.conv2g = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.batchNorm2g = nn.BatchNorm2d(64)
        self.fc_g = nn.Linear(64, num_classes)  

        
        self.conv1h_r = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3)
        self.batchNorm1h_r = nn.BatchNorm2d(64)
        self.conv2h_r = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3)
        self.batchNorm2h_r = nn.BatchNorm2d(64)
        self.fc_h_r = nn.Linear(64, 8)


        self.conv1h_g = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3)
        self.batchNorm1h_g = nn.BatchNorm2d(64)
        self.conv2h_g = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3)
        self.batchNorm2h_g = nn.BatchNorm2d(64)
        self.fc_h_g = nn.Linear(64, 8)


        self.conv1h_b = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3)
        self.batchNorm1h_b = nn.BatchNorm2d(64)
        self.conv2h_b = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size = 3)
        self.batchNorm2h_b = nn.BatchNorm2d(64)
        self.fc_h_b = nn.Linear(64, 8)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.soft = torch.nn.Softmax(dim = 1)
        

    def f_parameters(self):
        return list(self.conv1f.parameters()) + list(self.conv2f.parameters())

    def g_parameters(self):
        return list(self.conv1g.parameters()) + list(self.conv2g.parameters()) + list(self.fc_g.parameters())

    def h_parameters(self):
        return list(self.conv1h_r.parameters()) + list(self.conv2h_r.parameters()) + list(self.fc_h_r.parameters())+list(self.conv1h_g.parameters()) + list(self.conv2h_g.parameters()) + list(self.fc_h_g.parameters())+list(self.conv1h_b.parameters()) + list(self.conv2h_b.parameters()) + list(self.fc_h_b.parameters())

        
    def forward(self, img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))

        g = self.relu(self.conv1g(f))
        g = self.pool(self.relu(self.conv2g(g)))
        g = g.view(g.size(0), -1)
        g = self.fc_g(g)

        h = self.relu(self.conv1h(f))
        h = self.pool(self.relu(self.conv2h(h)))
        h = h.view(h.size(0), -1)
        h = self.fc_h(h)
        

        return g,h
        

    def predict_number(self,img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))

        g = self.relu(self.batchNorm1g(self.conv1g(f)))
        g = self.pool(self.relu(self.batchNorm2g(self.conv2g(g))))
        g = g.view(g.size(0), -1)
        g = self.fc_g(g)

        return g

    
    def predict_bias_inv_r(self,img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))
        
        f = GradientReversal.apply(f)

        h = self.relu(self.batchNorm1h_r(self.conv1h_r(f)))
        h = self.pool(self.relu(self.batchNorm2h_r(self.conv2h_r(h))))
        h = h.view(h.size(0), -1)
        h = self.fc_h_r(h)

        return h

    def predict_bias_r(self,img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))

        h = self.relu(self.batchNorm1h_r(self.conv1h_r(f)))
        h = self.pool(self.relu(self.batchNorm2h_r(self.conv2h_r(h))))
        h = h.view(h.size(0), -1)
        h = self.fc_h_r(h)

        return h

    def predict_bias_inv_g(self,img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))
        
        f = GradientReversal.apply(f)

        h = self.relu(self.batchNorm1h_g(self.conv1h_g(f)))
        h = self.pool(self.relu(self.batchNorm2h_g(self.conv2h_g(h))))
        h = h.view(h.size(0), -1)
        h = self.fc_h_g(h)

        return h

    def predict_bias_g(self,img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))

        h = self.relu(self.batchNorm1h_g(self.conv1h_g(f)))
        h = self.pool(self.relu(self.batchNorm2h_g(self.conv2h_g(h))))
        h = h.view(h.size(0), -1)
        h = self.fc_h_g(h)

        return h

    def predict_bias_inv_b(self,img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))
        
        f = GradientReversal.apply(f)

        h = self.relu(self.batchNorm1h_b(self.conv1h_b(f)))
        h = self.pool(self.relu(self.batchNorm2h_b(self.conv2h_b(h))))
        h = h.view(h.size(0), -1)
        h = self.fc_h_b(h)

        return h

    def predict_bias_b(self,img):
        f = self.pool(self.relu(self.batchNorm1f(self.conv1f(img))))
        f = self.pool(self.relu(self.batchNorm2f(self.conv2f(f))))

        h = self.relu(self.batchNorm1h_b(self.conv1h_b(f)))
        h = self.pool(self.relu(self.batchNorm2h_b(self.conv2h_b(h))))
        h = h.view(h.size(0), -1)
        h = self.fc_h_b(h)
        return h
        
        
        
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output
