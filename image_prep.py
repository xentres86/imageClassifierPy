import math
import numpy as np
from PIL import Image


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model

    Args:
        image: The image to be processed
    Returns:
        np_image: Processed image as numpy array
    """
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    size = 256, 256
    if im.width > im.height:
        ratio = float(im.width) / float(im.height)
        newheight = ratio * size[0]
        im = im.resize((size[0], int(math.floor(newheight))), Image.ANTIALIAS)
    else:
        ratio = float(im.height) / float(im.width)
        newwidth = ratio * size[0]
        im = im.resize((int(math.floor(newwidth)), size[0]), Image.ANTIALIAS)

    im = im.crop((im.width/2 - 112, im.height/2 - 112, im.width/2 + 112, im.height/2 + 112))
    
    np_image = np.array(im)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean)/std
    np_image =  np.transpose(np_image, (2, 0, 1))
    
    return np_image