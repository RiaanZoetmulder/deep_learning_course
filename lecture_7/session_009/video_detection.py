# Video object-detection demo – process MP4 / MOV and overlay bounding boxes.
# Compatible with any torchvision detection model (Faster R-CNN, FCOS, …).

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from IPython.display import HTML, display
from typing import List, Optional, Callable
from torchvision import transforms


# COCO 91-class names (index 0 = __background__)
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def _draw_boxes_on_frame(
    frame_rgb: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    score_thr: float = 0.5,
    box_color: tuple = (0, 255, 0),
    text_color: tuple = (255, 255, 255),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes and labels directly on an RGB frame (in-place copy)."""
    out = frame_rgb.copy()
    for box, lbl, sc in zip(boxes, labels, scores):
        if sc < score_thr:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)
        name = class_names[int(lbl)] if int(lbl) < len(class_names) else str(int(lbl))
        text = f"{name} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
    return out


def process_video_detection(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    preprocess: Optional[Callable] = None,
    score_thr: float = 0.45,
    target_fps: float = 10.0,
    max_frames: Optional[int] = None,
    figsize: tuple = (12, 7),
    verbose: bool = True,
) -> HTML:
    """Process a video through a detection model and return an HTML5 animation.

    Works with any torchvision-style detection model whose forward() returns
    ``[{"boxes": Tensor, "labels": Tensor, "scores": Tensor}, ...]``.

    Args:
        video_path:  Path to an MP4 / MOV / AVI file.
        model:       Detection model (already in eval mode).
        device:      ``torch.device`` for inference.
        class_names: List mapping label index → class name (default: COCO 91).
        preprocess:  Optional callable ``(ndarray H×W×3 RGB 0-255) → Tensor``.
                     If *None*, a standard ImageNet-normalised transform is used.
        score_thr:   Only draw boxes above this confidence.
        target_fps:  Target playback speed (frames per second).
        max_frames:  Cap the total processed frames (useful for long videos).
        figsize:     Matplotlib figure size ``(width, height)``.
        verbose:     Print progress messages.

    Returns:
        ``IPython.display.HTML`` containing an embedded HTML5 video.
    """
    if class_names is None:
        class_names = COCO_CLASSES

    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # uint8 HWC → float CHW [0,1]
        ])

    # --- Read video metadata ------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, int(round(fps / target_fps)))

    if verbose:
        print(f"Video: {video_path}")
        print(f"  {total} frames @ {fps:.1f} FPS, sampling every {sample_rate} → ~{total // sample_rate} frames")
        print(f"  Score threshold: {score_thr}")

    # --- Frame-by-frame inference -------------------------------------------
    frames_annotated: list[np.ndarray] = []
    model.eval()
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue
        if max_frames is not None and len(frames_annotated) >= max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess and run model
        img_t = preprocess(frame_rgb).to(device)
        with torch.no_grad():
            preds = model([img_t])[0]

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        annotated = _draw_boxes_on_frame(
            frame_rgb, boxes, labels, scores,
            class_names=class_names, score_thr=score_thr,
        )
        frames_annotated.append(annotated)
        frame_idx += 1

        if verbose and len(frames_annotated) % 50 == 0:
            print(f"  processed {len(frames_annotated)} frames …")

    cap.release()

    if not frames_annotated:
        raise RuntimeError("No frames were decoded – is the video file valid?")

    if verbose:
        print(f"✓ {len(frames_annotated)} frames ready for animation")

    # --- Build matplotlib animation -----------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    im = ax.imshow(frames_annotated[0])
    ax.set_title("Object Detection", fontsize=14)
    plt.tight_layout()

    def _update(i: int):
        im.set_array(frames_annotated[i])
        return [im]

    anim = animation.FuncAnimation(
        fig, _update,
        frames=len(frames_annotated),
        interval=1000 / target_fps,
        blit=True,
    )
    plt.close(fig)

    return HTML(anim.to_html5_video())


def display_video_detection(video_path: str, model: torch.nn.Module,
                            device: torch.device, **kwargs) -> None:
    """Convenience wrapper: process + display in one call."""
    html_video = process_video_detection(video_path, model, device, **kwargs)
    display(html_video)
