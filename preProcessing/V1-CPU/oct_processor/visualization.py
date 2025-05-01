import matplotlib.pyplot as plt

def show_edge_detection(original, binary_edges, overlay):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(binary_edges, cmap='gray')
    axes[1].set_title('Binary Edge Map')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Edges Overlaid')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def show_skeleton(original, skeleton_overlay):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(skeleton_overlay)
    ax.set_title('Skeleton Overlaid')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
