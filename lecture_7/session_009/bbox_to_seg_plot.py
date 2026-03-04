import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import numpy as np 

def bbox_to_seg_plot(image, VOC_COLORMAP, class_map, detected_bboxes):
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')

    # Panel 2: Segmentation mask (coloured)
    coloured_mask = VOC_COLORMAP[np.clip(class_map, 0, 20)]   # clip void to avoid index error
    coloured_mask[class_map == 255] = [224, 224, 192]           # void/border → light grey
    axes[1].imshow(coloured_mask)
    axes[1].set_title("Segmentation Mask", fontsize=14)
    axes[1].axis('off')

    # Panel 3: Original image with derived bounding boxes overlaid
    axes[2].imshow(image)
    for det in detected_bboxes:
        x1, y1, x2, y2 = det['bbox']
        colour = VOC_COLORMAP[det['class_id']] / 255.0  # normalise to [0, 1] for matplotlib
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=3, edgecolor=colour, facecolor='none')
        axes[2].add_patch(rect)
        axes[2].text(x1, y1 - 5, f"{det['class_name']}",
                    fontsize=11, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=colour, alpha=0.8))
    axes[2].set_title("Derived Bounding Box", fontsize=14)
    axes[2].axis('off')

    plt.suptitle("From Segmentation Mask to Bounding Box", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()