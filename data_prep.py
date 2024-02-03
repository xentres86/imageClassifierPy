import torch
from torchvision import datasets, models, transforms


def data_transformations():
    """Setup data transformations for train, valid and test Dataset
    
    Returns:
        data_transforms: Dict with transforms for train, valid and test datasets
    """
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])
        ])
    }
    
    return data_transforms

    
def data_images(train_dir, valid_dir, test_dir, data_transforms):
    """Setup datasets for images from the train, valid and test folder
    
    Args:
        train_dir: The Directory with train images
        valid_dir: The Directory with valid images
        test_dir: The Directory with test images
        data_transforms: Dict with transforms fro train, valid and test datasets
    Returns:
        image_datasets: The image datasets for train, valid and test
    """
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }
    
    return image_datasets


def data_loadings(image_datasets, batch_s):
    """Setup dataloaders for train, valid and test datasets
    
    Args:
        image_datasets: The image datasets for train, valid and test
        batch_s: The batch size of the dataloader
    Returns:
        dataloaders: The dataloaders for train, valid and test
    """
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_s, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_s),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_s)
    }
    
    return dataloaders