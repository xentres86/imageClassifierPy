import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import models
from collections import OrderedDict


def setup_pretrained_model_with_classifier(model_type, hidden_units, output_units, drop_p):
    """Setup the pretrained Model Type (VGG/DENSENET) with custom classifier
    
    Args:
        model_type: The model_type for the pretrained network (VGG/DENSENET)
        hidden_units: The number of hidden units in the hidden layers
        output_units: The number of output units in the output layer
        drop_p: Dropout percentage of the network
    Returns:
        model: The pretrained model with the custom classifier
    """
    if model_type == 'VGG':
        model = models.vgg16(pretrained=True)
        in_units = 25088
    elif model_type == 'DENSENET':
        model = models.densenet121(pretrained=True)
        in_units = 1024

    # Fix features
    for param in model.parameters():
        param.requires_grad = False
    
    # Setup Network classifier
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_units, hidden_units)),
                              ('relu1', nn.ReLU(True)),
                              ('drop1', nn.Dropout(drop_p)),
                              ('fc2', nn.Linear(hidden_units, hidden_units)),
                              ('relu2', nn.ReLU(True)),
                              ('fc3', nn.Linear(hidden_units, output_units)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Set new classifier
    model.classifier = classifier
    
    return model


def setup_network_optimization_parameters(model, learning_rate, moment):
    """Setup network optimization parameters (criterion, optimizer, scheduler)
    
    Args:
        model: The pretrained model with the custom classifier
        learning_rate: The learning rate of the network
        moment: The momentum of the network
    Returns:
        criterion: The criterion with CrossEntropyLoss
        optimizer: The Optimizer with SGD, learning rate and momentum
        scheduler: The scheduler with step size and learning rate gamma
    """
    # Set criterion, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=moment)
    
    # Scheduler configuration
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    return criterion, optimizer, scheduler