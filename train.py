import argparse
import time
import copy
import torch
import device_setup
import network_prep
import data_prep
from checkpoint_setup import save_checkpoint
from workspace_utils import active_session


def argument_parser():
    """Argument Parser for parsing Argument from the command line
    
    Args:
        data_dir: The directory with the image data
        save_dir: Save directory of a Checkpoint
        arch: Architecture fo the Neural Network
        learning_rate: Learningrate of the Network
        moment: Moment of the Network
        hidden_units: Hidden Units of the classifier
        output_units: Output Units of the classifier
        drop_p: Dropout percentage of the classifier  
        epochs: Epochs to learn
        batch_s: Batch size for dataloaders
        gpu: Devicemode
    Returns:
        parser.parse_args(): The argparse.namespace
    """
    parser = argparse.ArgumentParser(description='Input for Training')

    parser.add_argument('data_dir', type=str, metavar='data_dir', help='The directory with the image data')
    parser.add_argument('--save_dir', type=str, default= 'my_trained_model_checkpoint.pth', help='Save directory of a checkpoint')
    parser.add_argument('--arch', type=str, default= 'VGG', choices=['VGG', 'DENSENET'], help='Architecture of the Neural Network')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learningrate of the Network')
    parser.add_argument('--moment', type=float, default=0.9, help='Momentum of the Network')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden Units of the classifier')
    parser.add_argument('--output_units', type=int, default=102, help='Output Units of the classifier')
    parser.add_argument('--drop_p', type=float, default=0.5, help='Dropout percentage of the classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs to learn')
    parser.add_argument('--batch_s', type=int, default=8, help='Batch size for dataloaders')
    parser.add_argument('--gpu', default=False, action='store_true', help='Devicemode')

    return parser.parse_args()


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, epochs):
    """Train the model
    
    Args:
        model: The pretrained model with the custom classifier
        dataloaders: The dataloader for the train, valid and test dataset
        dataset_sizes: Sizes of the datasets
        criterion: The criterion with CrossEntropyLoss
        optimizer: The Optimizer with SGD, learning rate and momentum
        scheduler: The scheduler with step size and learning rate gamma
        device: The device mode (GPU/CPU)
        epochs: Number of epochs to learn
    Returns:
        model: The trained model
    """
    since = time.time()
    
    # Make initial deepcopy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Change to device
    model.to(device)

    with active_session():
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    return model


def test_model(model, dataloaders, criterion, device):
    """Test the model
    
    Args:
        model: The pretrained model with the custom classifier
        dataloaders: The dataloader for the train, valid and test dataset
        criterion: The criterion with CrossEntropyLoss
        device: The device mode (GPU/CPU)
    """
    model.to(device)
    model.eval()

    accuracy = 0
    test_loss = 0
    
    for ii, (images, labels) in enumerate(dataloaders['test']):

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).data.item()

        # Calculating the accuracy 
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    print("Batch: {} ".format(ii+1),
          "Test Loss: {:.3f}.. ".format(test_loss/len(dataloaders['test'])),
          "Test Accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))

    
def main(args):
    """The main Function"""
    # Check data in argparse
    print(args)
    
    # Set Directorys
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Set device mode to GPU oder CPU
    device = device_setup.set_device_mode(args.gpu)
    
    # Initialise transform, dataset and dataloader
    data_transforms = data_prep.data_transformations()
    image_datasets = data_prep.data_images(train_dir, valid_dir, test_dir, data_transforms)
    dataloaders = data_prep.data_loadings(image_datasets, args.batch_s)
    
    # Check Data in transfom, dataset and dataloader
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    print ("Dasaset Size: "+ str(dataset_sizes))
      
    # Set the pretrained model
    model = network_prep.setup_pretrained_model_with_classifier(args.arch, args.hidden_units, args.output_units, args.drop_p)
    # Show Model Setup
    print(model)
    
    # Set network optimization parameters
    criterion, optimizer, scheduler = network_prep.setup_network_optimization_parameters(model, args.learning_rate, args.moment)
    
    # Train the model
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, args.epochs)
    
    # Test the model
    test_model(model, dataloaders, criterion, device)
    
    # Save the model
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    save_checkpoint(model, args.arch, criterion, optimizer, args.drop_p, args.batch_s, args.epochs, args.learning_rate, args.moment, args.save_dir)
    print("Train, test and save completed!")


if __name__ == "__main__":
    """main sentinel"""
    main(argument_parser())