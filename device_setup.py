import torch


def set_device_mode(gpu):
    """Set device mode to cuda or cpu
    
    Args:
        gpu: Gpu mode true or false
    Returns:
        device: Device mode
    """
    if gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            print("Cuda not available, returning to CPU")
    else:
        device = 'cpu'
    print("Device set to " + device)
    
    return device