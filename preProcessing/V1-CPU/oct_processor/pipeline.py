from .preprocessing import load_and_prepare_image
from .shearlet_utils import get_shearlet_coeffs
from .edge_detection import extract_edges
from .flattening import flatten_image
from .postprocessing import postprocess_edges
from .visualization import show_edge_detection, show_skeleton
import matplotlib.pyplot as plt
import numpy as np

def process_oct_image(
    file_path,
    device,
    qLen=512,
    denoise_sigma=0.08,
    nScales1=2,
    shearLevels1=[1, 1],
    reconScales1=[1],
    T1=1e-20,
    epsilon1=0.35,
    flatten_mode='crop',
    poly_deg=4,
    padding_top=100,
    padding_bot=50,
    flatten_sigma=0.1,
    nScales2=2,
    shearLevels2=[1,2],
    reconScales2=[1,2],
    T2=1e-20,
    epsilon2=0.28,
    footprint=[2,8],
    edge_threshold=0.8,
    visualize=False
):

    image = load_and_prepare_image(file_path, qLen, device, denoise_sigma)
    even_coeffs, odd_coeffs = get_shearlet_coeffs(image, nScales=nScales1, shearLevels=shearLevels1)
    edge_map = extract_edges(even_coeffs, odd_coeffs, 2, reconScales1, epsilon1, T1)

    binary_edges = edge_map > edge_threshold

    flat_image, polyCoeffs, cut_top, cut_bottom, reference_row = flatten_image(
        binary_edges, image, reference_row=qLen//2, poly_deg=poly_deg,
        padding_top=padding_top, padding_bot=padding_bot, flatten_mode=flatten_mode,
        sigma=flatten_sigma
    )
  
    even2, odd2 = get_shearlet_coeffs(flat_image, nScales=nScales2, shearLevels=shearLevels2)
    edge_map2 = extract_edges(even2, odd2, 2, reconScales2, epsilon2, T2)

    binary_edges_2 = edge_map2 > edge_threshold

    binary_edges, skeleton, overlay, skeleton_overlay = postprocess_edges(
        binary_edges_2, flat_image, threshold=edge_threshold, footprint_shape=footprint
    )

    if visualize:
        show_edge_detection(flat_image, binary_edges, overlay)
        show_skeleton(flat_image, skeleton_overlay)

    return skeleton, edge_map2, flat_image




   
