import numpy as np
from skimage.transform import resize
from bm3d import bm3d, BM3DProfile
import matplotlib.pyplot as plt


def flatten_image(binary_edges, image, reference_row=256, poly_deg=4,
                  padding_top=100, padding_bot=200, flatten_mode='crop', sigma=0.1):
    nx, ny = binary_edges.shape
    x_coords, y_coords, y_min_coords, y_max_coords = [], [], [], []

    for x in range(ny):
        edge_rows = np.where(binary_edges[:, x])[0]
        if edge_rows.size > 0:
            y_coords.append(edge_rows[len(edge_rows) // 2])
            y_min_coords.append(edge_rows[0])
            y_max_coords.append(edge_rows[-1])
            x_coords.append(x)

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    y_min_coords = np.array(y_min_coords)
    y_max_coords = np.array(y_max_coords)

    x_full = np.arange(ny)
    y_coords_full = np.interp(x_full, x_coords, y_coords)
    y_min_full = np.interp(x_full, x_coords, y_min_coords)
    y_max_full = np.interp(x_full, x_coords, y_max_coords)

    polyCoeffs = np.polyfit(x_full, y_coords_full, deg=poly_deg)



    flattened = np.zeros_like(image)
    for x in range(ny):
        y_boundary = int(round(np.polyval(polyCoeffs, x)))
        shift = reference_row - y_boundary
        flattened[:, x] = np.roll(image[:, x], shift)
    
    # plt.figure(figsize=(5, 5))
    # plt.imshow(flattened, cmap='gray')
    # plt.title("Just Flattened (before crop/pad)")
    # plt.show()

    dx_min = y_min_full - y_coords_full
    dx_max = y_max_full - y_coords_full

    avg_dx_min = int(np.mean(dx_min))
    avg_dx_max = int(np.mean(dx_max))

    cut_top = max(0, min(reference_row + avg_dx_min - padding_top, nx - 1))
    # cut_top = reference_row + avg_dx_min - padding_top
    cut_bottom = max(cut_top + 1, min(reference_row + avg_dx_max + padding_bot, nx))
    # cut_bottom = reference_row + avg_dx_max + padding_bot
    
    # plt.figure(figsize=(5, 5))
    # plt.imshow(image, cmap='gray')
    # plt.plot(x_full, np.polyval(polyCoeffs, x_full), 'r-', label='Middle Boundary')
    # plt.plot(x_full, np.polyval(polyCoeffs, x_full) + avg_dx_min - padding_top, 'y-', label='Upper Boundary')
    # plt.plot(x_full, np.polyval(polyCoeffs, x_full) + avg_dx_max + padding_bot, 'b-', label='Lower Boundary')
    # plt.axis('off')
    # plt.title("Original Image (before crop/pad)")
    # plt.show()

    if flatten_mode == 'pad':
        cut_image = flattened[max(cut_top, 0):min(cut_bottom, nx), :]
        target_height = nx
        current_height = cut_image.shape[0]
        pad_total = max(0, target_height - current_height)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        final_image = np.pad(cut_image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)
    elif flatten_mode == 'crop':
        cut_image = flattened[min(cut_top, cut_bottom):max(cut_top, cut_bottom), :]
        final_image = resize(cut_image, (nx, ny), order=1, preserve_range=True, anti_aliasing=True)
    else:
        raise ValueError(f"Unknown flatten_mode: {flatten_mode}. Use 'pad' or 'crop'.")
    

    final_image_filt = bm3d(final_image, sigma, profile=BM3DProfile())
    return final_image_filt, polyCoeffs, cut_top, cut_bottom, reference_row
