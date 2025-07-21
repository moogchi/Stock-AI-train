import torch
import os

def save_model(model, path):
    """
    Save the PyTorch model to the given path.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): File path to save the model.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, device=None, **model_kwargs):
    """
    Load model weights into an instance of model_class.

    Args:
        model_class (class): The class of the model architecture.
        path (str): Path to the saved model weights.
        device (torch.device, optional): Device to map the model to. Defaults to CPU if None.
        **model_kwargs: Keyword args to instantiate the model.

    Returns:
        torch.nn.Module: The loaded model.
    """
    if device is None:
        device = torch.device('cpu')

    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")
    return model
