# Video segmentation visualization utilities
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
from typing import Optional


def process_video_segmentation(
    video_path: str,
    model: torch.nn.Module,
    colormap: np.ndarray,
    device: torch.device,
    mean_bgr: np.ndarray = np.array([104.00698793, 116.66876762, 122.67891434]),
    target_fps: float = 10.0,
    alpha: float = 0.5,
    figsize: tuple = (15, 5),
    verbose: bool = True
) -> HTML:
    """
    Process a video through a segmentation model and display as an animation.
    
    Args:
        video_path: Path to the input video file
        model: PyTorch segmentation model (should output class logits)
        colormap: Numpy array of shape (num_classes, 3) mapping class indices to RGB colors
        device: Torch device to run inference on
        mean_bgr: BGR mean values for preprocessing (default: VGG/FCN values)
        target_fps: Target frames per second for display (default: 10)
        alpha: Blend factor for overlay (0=original, 1=segmentation)
        figsize: Figure size for the animation (width, height)
        verbose: Whether to print progress messages
        
    Returns:
        HTML object containing the animation video
    """
    if verbose:
        print('Rendering this takes 2 or so minutes on my computer. Once it is done, press the play button on the video to see the results!')
    
    # Read video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sample rate to achieve target fps
    sample_rate = max(1, int(fps // target_fps))
    
    frames_original = []
    frames_segmented = []
    frames_overlay = []
    
    model.eval()
    
    if verbose:
        print(f"Processing frames for display (sampling every {sample_rate} frames)...")
    
    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB for matplotlib
            
            # Inference
            frame_input = frame_bgr.astype(np.float32) - mean_bgr
            frame_tensor = torch.from_numpy(frame_input.transpose(2, 0, 1).copy()).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                output = model(frame_tensor)
                pred = output.argmax(dim=1).squeeze().cpu().numpy()
            
            pred_colored = colormap[pred]  # Map predictions to RGB colors
            
            # Create overlay (blend original with segmentation)
            overlay = (frame_rgb * (1 - alpha) + pred_colored * alpha).astype(np.uint8)
            
            frames_original.append(frame_rgb)
            frames_segmented.append(pred_colored)
            frames_overlay.append(overlay)
        
        frame_idx += 1
    
    cap.release()
    
    if verbose:
        print(f"✓ Prepared {len(frames_original)} frames for animation")
    
    # Create animation
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax in axes:
        ax.axis('off')
    
    im0 = axes[0].imshow(frames_original[0])
    im1 = axes[1].imshow(frames_segmented[0])
    im2 = axes[2].imshow(frames_overlay[0])
    axes[0].set_title('Original')
    axes[1].set_title('Segmentation')
    axes[2].set_title('Overlay')
    
    def animate(i):
        im0.set_array(frames_original[i])
        im1.set_array(frames_segmented[i])
        im2.set_array(frames_overlay[i])
        return [im0, im1, im2]
    
    anim = animation.FuncAnimation(
        fig, animate, 
        frames=len(frames_original), 
        interval=1000 / target_fps, 
        blit=True
    )
    plt.tight_layout()
    plt.close()
    
    # Return HTML5 video
    return HTML(anim.to_html5_video())


def display_video_segmentation(
    video_path: str,
    model: torch.nn.Module,
    colormap: np.ndarray,
    device: torch.device,
    **kwargs
) -> None:
    """
    Process and display a video segmentation animation.
    
    This is a convenience wrapper around process_video_segmentation that
    automatically displays the result.
    
    Args:
        video_path: Path to the input video file
        model: PyTorch segmentation model
        colormap: Class index to RGB color mapping
        device: Torch device for inference
        **kwargs: Additional arguments passed to process_video_segmentation
    """
    html_video = process_video_segmentation(
        video_path=video_path,
        model=model,
        colormap=colormap,
        device=device,
        **kwargs
    )
    display(html_video)
