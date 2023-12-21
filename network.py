import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, out_channels_list=[16, 32], hidden_dims=[256, 128]):
        super(ConvNet, self).__init__()
        channel_1 = out_channels_list[0]
        channel_2 = out_channels_list[1]
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=channel_1, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=channel_1, out_channels=channel_2, kernel_size=3, stride=1, padding=0),
                                   nn.ReLU(), 
                                   nn.AvgPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Linear(channel_2*6*6, hidden_dims[0]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dims[0], hidden_dims[1]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dims[1], 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  
        x = x.view(x.size(0), -1) 
        output = self.classifier(x) 

        return output
