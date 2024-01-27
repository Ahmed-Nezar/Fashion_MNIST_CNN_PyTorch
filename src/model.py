from load_data import *


class ConvNet (nn.Module):
    def __init__(self,in_channels = 1, out_channels = 32):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fcn1 = nn.Linear(out_channels*13*13, 50)
        torch.nn.init.kaiming_uniform(self.fcn1.weight, nonlinearity='relu')
        self.bn1 = nn.BatchNorm1d(50)
        self.fcn2 = nn.Linear(50, 10)
        torch.nn.init.kaiming_uniform(self.fcn2.weight, nonlinearity='relu')
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = x.view(x.shape[0], -1)
        x = self.fcn1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fcn2(x)
        return x