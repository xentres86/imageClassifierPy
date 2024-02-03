import argparse
import json
import torch
import numpy
import device_setup
import image_prep
from checkpoint_setup import load_checkpoint


def argument_parser():
    """Argument Parser for parsing Argument from the command line
    
    Args:
        data_dir: The directory with the image data
        load_dir: Load directory of a Checkpoint
        top_k: Top k most likely classes
        category_names: Mapping of categories to real names
        gpu: Devicemode
    Returns:
        parser.parse_args(): The argparse.namespace
    """
    parser = argparse.ArgumentParser(description='Input for Training')

    parser.add_argument('data_dir', type=str, metavar='data_dir', help='The directory with the image data')
    parser.add_argument('--load_dir', type=str, default= 'my_trained_model_checkpoint.pth', help='Load directory of a Checkpoint')
    parser.add_argument('--top_k', type=int, default= 5, help='Top k most likely classes')
    parser.add_argument('--category_names', type=str, default= 'cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', default=False, action='store_true', help='Devicemode')

    return parser.parse_args()


def predict(image_path, model, device, category_n, class_to_idx, topk):
    """Predict the class (or classes) of an image using a trained deep learning model

    Args:
        image_path: The path to the image
        model: The pretrained model
        device: Device mode of the network
        topk: Number of top predictions
    Returns:
        topk number of names and indexes of prediction
    """

    # Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    
    image = torch.FloatTensor([image_prep.process_image(image_path)])
    
    with torch.no_grad():
        output = model.forward(image.to(device))
    
    ps = torch.exp(output)
    
    # Save probs and classes
    probs = ps.topk(topk)[0][0].cpu().numpy()
    classes = ps.topk(topk)[1][0].cpu().numpy()
    
    # Set the probs value to percent value
    probs_perc = [x*100 for x in probs]
    
    # Map the predicted class numbers and coresponding class names
    with open(category_n, 'r') as f:
        cat_to_name = json.load(f)
        
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    
    predicted_classes_num = [idx_to_class[x] for x in classes]
    predicted_classes_name = [cat_to_name[x] for x in predicted_classes_num]
    
    return probs_perc, predicted_classes_num, predicted_classes_name


def main(args):
    """The main Function"""
    # Check data in argparse
    print(args)
    
    # Set device mode to GPU oder CPU
    device = device_setup.set_device_mode(args.gpu)
    
    # Load the pretrained model with criterion, optimizer and class to idx mapping
    loaded_model, loaded_criterion, loaded_optimizer, class_to_idx = load_checkpoint(args.load_dir)

    # Predict the class of the image
    probs_perc, predicted_classes_num, predicted_classes_name = predict(args.data_dir, loaded_model, device, args.category_names, class_to_idx, args.top_k)
    
    # Print the results
    for i in range(len(predicted_classes_name)):
        print("{}. - ClassNum: {} - ClassName: {} - {:.2f}%".format(i+1, predicted_classes_num[i], predicted_classes_name[i], probs_perc[i]))

if __name__ == "__main__":
    """main sentinel"""
    main(argument_parser())