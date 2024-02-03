import torch
from torchvision import models


def save_checkpoint(model, arch, criterion, optimizer, drop_p, batch_s, epochs, learning_r, moment, save_dir):
    """Save the model
    
    Args:
        model: The pretrained model with the custom classifier
        arch: The architecture of the pretrained model
        criterion: The criterion with CrossEntropyLoss
        optimizer: The Optimizer with SGD, learning rate and momentum
        drop_p: Dropout percentage of the network
        batch_s: Batch size of the dataloader
        epochs: Epochs for network training
        learning_r: Learning rate of the network
        moment: Momentum of the network
        save_dir: The save directory of the model checkpoint
    """
    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'dropout': drop_p,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'criterion': criterion,
        'class_to_idx': model.class_to_idx,
        'batch_size': batch_s,
        'epochs': epochs,
        'learning_rate': learning_r,
        'momentum': moment
    }

    torch.save(checkpoint, save_dir)
    print("Checkpoint " + save_dir + " saved!")
    

def load_checkpoint(filepath):
    """Load the model
    
    Args:
        filepath: The path to the saved checkpoint
    Returns:
        mode: The loaded model
        criterion: The criterion of the model
        optimizer_dict: The optimizer state dict of the model
        class_to_idx: The class_to_idx of the model
    """
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'VGG':
        model = models.vgg16()
    elif checkpoint['arch'] == 'DENSENET':
        model = models.densenet121()
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['criterion'], checkpoint['optimizer_dict'], checkpoint['class_to_idx']