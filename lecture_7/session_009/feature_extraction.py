# Feature extraction utilities for transfer learning demonstrations
import torch
import numpy as np
from typing import Tuple, List, Callable
from torch.utils.data import Dataset


def extract_vgg_features(
    dataset: Dataset,
    pretrained_model: torch.nn.Module,
    untrained_model: torch.nn.Module,
    device: torch.device,
    num_examples: int = 400,
    print_interval: int = 50,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Extract feature maps from pretrained and untrained VGG16 models for comparison.
    
    This function iterates over a dataset and extracts features from both models,
    useful for visualizing the effect of transfer learning (e.g., with t-SNE).
    
    Args:
        dataset: PyTorch Dataset that returns (image, label) tuples
        pretrained_model: VGG16 model with pretrained weights (returns features directly)
        untrained_model: VGG16 model with random weights (returns output, skip_connections)
        device: Torch device for inference
        num_examples: Maximum number of examples to process
        print_interval: Print progress every N examples
        verbose: Whether to print progress messages
        
    Returns:
        Tuple of:
            - feature_maps_pretrained: List of numpy arrays with pretrained features
            - feature_maps_untrained: List of numpy arrays with untrained features  
            - labels: List of integer labels
            
    Example:
        >>> pretrained_features, untrained_features, labels = extract_vgg_features(
        ...     dataset=voc_dataset,
        ...     pretrained_model=vgg16_pretrained,
        ...     untrained_model=vgg16_untrained,
        ...     device=device,
        ...     num_examples=400
        ... )
    """
    # Set models to evaluation mode
    pretrained_model.eval()
    untrained_model.eval()
    
    feature_maps_pretrained = []
    feature_maps_untrained = []
    labels = []
    
    # Determine how many examples to process
    n_samples = min(num_examples, len(dataset))
    
    if verbose:
        print(f"Extracting features from {n_samples} examples...")
        print("This takes about 1-2 minutes depending on hardware.")
    
    for i in range(n_samples):
        image, label = dataset[i]
        labels.append(label)
        
        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(device)
        
        # Turn off gradients for inference
        with torch.no_grad():
            # Get features from pretrained model
            feature_map = pretrained_model(image)
            
            # Get features from untrained model (assumes it returns output, skip_connections)
            _, skip_connections_untrained = untrained_model(image)
            
            # Store the last skip connection (deepest features before FC layers)
            feature_maps_untrained.append(skip_connections_untrained[-1].cpu().numpy())
            feature_maps_pretrained.append(feature_map.cpu().numpy())
        
        if verbose and i % print_interval == 0:
            print(f"Processed {i}/{n_samples} examples")
    
    if verbose:
        print(f"Done! Extracted features from {len(feature_maps_pretrained)} images")
    
    return feature_maps_pretrained, feature_maps_untrained, labels
