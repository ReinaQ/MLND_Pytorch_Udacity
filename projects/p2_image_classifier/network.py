import torch
from torch import nn
from torchvision import models
from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, arch='resnet152', hidden_units=512, pretrained=True):
        ''' Builds a feedforward network with selected pre-trained model and arbitrary hidden units.
        
        Arguments
        ---------
        arch: string, the name of pre-trained model (limited to resnet152 and densenet169)
        hidden_units: int, the number of the hidden units
        pretrained: bool, whether load the pre-trained parameters for the arch
        '''
        super().__init__()
        # Get pre-trained model 
        self.model = getattr(models, arch)(pretrained=pretrained)
        self.arch = arch
        self.hidden_units = hidden_units
        
        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Define the fully connected layer to fit dataset
        if arch =='resnet152':
            fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('Dropout1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

            # Attach the defined fc layer to pre-trained model arch
            self.model.fc = fc
        elif arch == 'densenet169':
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1664, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('Dropout1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
            # Attach the classifier to desnet model 
            self.model.classifier = classifier
            
    def forward(self, inputs):
        return self.model.forward(inputs)
    
    def to_device(self, device):
        self.model = self.model.to(device)
        
