import numpy as np
from skimage.morphology import binary_closing, skeletonize, thin, footprint_rectangle


def postprocess_edges(edge_map, image, threshold=0.8, footprint_shape=(2, 8)):
    binary_edges = edge_map > threshold
    thinned = thin(binary_edges)
    for _ in range(3):
        thinned = thin(thinned)
    footprint = footprint_rectangle(footprint_shape)
    closed_edges = binary_closing(thinned, footprint=footprint)
    skeleton = skeletonize(closed_edges)

    overlay = np.dstack([image]*3)
    overlay[thinned] = [1, 0, 0]  # red

    skeleton_overlay = overlay.copy()
    skeleton_overlay[skeleton] = [0, 1, 0]  # green

    return binary_edges, skeleton, overlay, skeleton_overlay
