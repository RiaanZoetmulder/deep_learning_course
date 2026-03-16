# --- Visualise a simple graph with three connected components ---
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import label


def visualize_connected_components():
    """
    Visualize 3 simple connected components. 
    """
    G = nx.Graph()

    # Component 1: A-B-C
    G.add_edges_from([('A', 'B'), ('A', 'C')])
    # Component 2: E-F-G
    G.add_edges_from([('E', 'F'), ('E', 'G')])
    # Component 3: D (isolated)
    G.add_node('D')

    # Fixed positions so the layout matches the conceptual diagram
    pos = {
        'A': (0, 1), 'B': (1, 1), 'C': (0, 0),
        'E': (3, 1), 'F': (4, 1), 'G': (3, 0),
        'D': (1.5, -0.2),
    }

    # Colour each component differently
    component_colours = {
        'A': '#1f77b4', 'B': '#1f77b4', 'C': '#1f77b4',   # blue
        'E': '#ff7f0e', 'F': '#ff7f0e', 'G': '#ff7f0e',   # orange
        'D': '#2ca02c',                                      # green
    }
    node_colors = [component_colours[n] for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    nx.draw(
        G, pos, ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=800,
        font_size=14,
        font_weight='bold',
        font_color='white',
        edge_color='#555555',
        width=2.5,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color='#1f77b4', label='Component 1: {A, B, C}'),
        mpatches.Patch(color='#ff7f0e', label='Component 2: {E, F, G}'),
        mpatches.Patch(color='#2ca02c', label='Component 3: {D}  (isolated)'),
    ]
    ax.legend(handles=legend_handles, loc='lower center', ncol=3,
            fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.12))

    ax.set_title('An undirected graph with three connected components', fontsize=12)
    plt.subplots_adjust(bottom=0.18)
    plt.show()


def visualize_2d_connected_components():
    """
    Visualise connected components in a simple 2D binary mask using both 4-connectivity and 8-connectivity.
    """
    # Create the small 6×6 binary mask from the explanation above
    binary_mask = np.array([
        [0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    # 4-connectivity structuring element (cross / plus shape)
    struct_4 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])

    # 8-connectivity structuring element (full 3×3 block)
    struct_8 = np.ones((3, 3), dtype=int)

    label_map_4, num_4 = label(binary_mask, structure=struct_4)
    label_map_8, num_8 = label(binary_mask, structure=struct_8)

    # --- Visualise ---
    cmap_labels = ListedColormap(['black', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Original binary mask
    axes[0].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Binary Mask (input)')
    for r in range(binary_mask.shape[0]):
        for c in range(binary_mask.shape[1]):
            axes[0].text(c, r, str(binary_mask[r, c]), ha='center', va='center',
                        color='white' if binary_mask[r, c] == 0 else 'black', fontsize=12)

    # 4-connected label map
    axes[1].imshow(label_map_4, cmap=cmap_labels, vmin=0, vmax=4)
    axes[1].set_title(f'4-Connectivity \u2192 {num_4} components')
    for r in range(label_map_4.shape[0]):
        for c in range(label_map_4.shape[1]):
            axes[1].text(c, r, str(label_map_4[r, c]), ha='center', va='center',
                        color='white', fontsize=12)

    # 8-connected label map
    axes[2].imshow(label_map_8, cmap=cmap_labels, vmin=0, vmax=4)
    axes[2].set_title(f'8-Connectivity \u2192 {num_8} components')
    for r in range(label_map_8.shape[0]):
        for c in range(label_map_8.shape[1]):
            axes[2].text(c, r, str(label_map_8[r, c]), ha='center', va='center',
                        color='white', fontsize=12)

    for ax in axes:
        ax.set_xticks(range(binary_mask.shape[1]))
        ax.set_yticks(range(binary_mask.shape[0]))
        ax.grid(True, color='gray', linewidth=0.5)

    plt.tight_layout()
    plt.show()