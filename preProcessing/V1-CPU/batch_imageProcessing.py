import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import pandas as pd

# python batch_edge_detection.py --input /path/to/data --table metadata.csv --n_threads 8

# Import your actual processing function here
from oct_processor import process_oct_image

def get_device_params(device):
    if device == "PlexElite":
        denoise_sigma = 0.1
        nScales1 = 2
        shearLevels1 = [1, 2]
        reconScales1 = [1, 2]
        T1 = 1e-20
        epsilon1 = 0.29
        flatten_mode = 'crop'
        poly_deg = 4
        padding_top = 100
        padding_bot = 200
        flatten_sigma = 0.01
        nScales2 = 2
        shearLevels2 = [1, 2, 3]
        reconScales2 = [1, 2]
        T2 = 1e20
        epsilon2 = 0.35
        footprint = [4, 15]
    elif device == "Heidelberg":
        denoise_sigma = 0.05
        nScales1 = 2
        shearLevels1 = [1, 1]
        reconScales1 = [1, 2]
        T1 = 1e20
        epsilon1 = 0.15
        flatten_mode = 'crop'
        poly_deg = 1
        padding_top = 100
        padding_bot = 200
        flatten_sigma = 0.0000000005
        nScales2 = 2
        shearLevels2 = [1, 2]
        reconScales2 = [1, 2]
        T2 = 1e50
        epsilon2 = 0.1
        footprint = [2, 15]
    elif device == "Cirrus":
        denoise_sigma = 0.1
        nScales1 = 2
        shearLevels1 = [1, 1]
        reconScales1 = [1, 2]
        T1 = 1e20
        epsilon1 = 0.3
        flatten_mode = 'crop'
        poly_deg = 4
        padding_top = 100
        padding_bot = 150
        flatten_sigma = 0.01
        nScales2 = 2
        shearLevels2 = [1, 2, 3]
        reconScales2 = [1, 2]
        T2 = 1e50
        epsilon2 = 0.25
        footprint = [3, 20]

    return (denoise_sigma, nScales1, shearLevels1, reconScales1, T1, epsilon1, flatten_mode, poly_deg, padding_top, padding_bot, flatten_sigma, nScales2, shearLevels2, reconScales2, T2, epsilon2, footprint)

# --- Configurable Preprocessing Wrapper ---
def process_image(img_path, params):
    try:
        device = "Cirrus"
        (denoise_sigma, nScales1, shearLevels1, reconScales1, T1, epsilon1,
         flatten_mode, poly_deg, padding_top, padding_bot, flatten_sigma,
         nScales2, shearLevels2, reconScales2, T2, epsilon2, footprint) = params['device_params']

        # Run Preprocessing
        skeleton, edge_map, flat_image = process_oct_image(
            file_path=img_path,
            device=device,
            denoise_sigma=denoise_sigma,
            nScales1=nScales1,
            shearLevels1=shearLevels1,
            reconScales1=reconScales1,
            T1=T1,
            epsilon1=epsilon1,
            flatten_mode=flatten_mode,
            poly_deg=poly_deg,
            padding_top=padding_top,
            padding_bot=padding_bot,
            flatten_sigma=flatten_sigma,
            nScales2=nScales2,
            shearLevels2=shearLevels2,
            reconScales2=reconScales2,
            T2=T2,
            epsilon2=epsilon2,
            footprint=footprint,
            edge_threshold=params['threshold'],
            visualize=False
        )

        if "B-Scans" not in img_path:
            raise ValueError("'B-Scans' not found in image path")

        # Define output paths
        base_path = os.path.dirname(os.path.dirname(img_path))
        filename = os.path.splitext(os.path.basename(img_path))[0]

        edge_map_path = os.path.join(base_path, "Edge-Map", filename + "_edge.png")
        flat_image_path = os.path.join(base_path, "Flattened", filename + "_flat.png")

        os.makedirs(os.path.dirname(edge_map_path), exist_ok=True)
        os.makedirs(os.path.dirname(flat_image_path), exist_ok=True)

        cv2.imwrite(edge_map_path, (skeleton * 255).astype(np.uint8))
        cv2.imwrite(flat_image_path, (flat_image * 255).astype(np.uint8))

        return img_path, True, ""
    except Exception as e:
        return img_path, False, str(e)

# --- Dataset-Based Image Collector ---
def collect_images_from_dataset(image_root_dir, dataframe):
    image_paths = []

    if dataframe is not None:
        for patient_id in os.listdir(image_root_dir):
            try:
                patient_id_int = int(patient_id)
                if dataframe['Patient Number'].isin([patient_id_int]).any():
                    patient_df = dataframe[dataframe['Patient Number'] == patient_id_int]
                    patient_path = os.path.join(image_root_dir, patient_id)
                    if not os.path.isdir(patient_path):
                        continue

                    for root, dirs, files in os.walk(patient_path):
                        if os.path.basename(root) == "B-Scans":
                            for img_name in files:
                                if img_name.lower().endswith(('.jpg', '.png')):
                                    image_paths.append(os.path.join(root, img_name))
            except (ValueError, TypeError):
                continue
    else:
        for root, _, files in os.walk(image_root_dir):
            if os.path.basename(root) == "B-Scans":
                for img_name in files:
                    if img_name.lower().endswith(('.jpg', '.png')):
                        image_paths.append(os.path.join(root, img_name))

    return image_paths


# --- Main Runner ---
def main(input_dir, table_path, num_workers):
    dataframe = pd.read_csv(table_path) if table_path else None
    image_paths = collect_images_from_dataset(input_dir, dataframe)
    print(f"Found {len(image_paths)} images. Starting batch processing...")

    device = "Cirrus"
    params = {
        'threshold': 0.2,
        'device_params': get_device_params(device)
    }

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(image_paths), desc="Processing Images") as pbar:
            worker_fn = partial(process_image, params=params)
            for img_path, success, message in pool.imap_unordered(worker_fn, image_paths):
                if not success:
                    tqdm.write(f"Failed: {img_path} | Error: {message}")
                pbar.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input image root folder")
    parser.add_argument("--table", type=str, default=None, help="CSV path to patient metadata table (optional)")
    parser.add_argument("--n_threads", type=int, default=cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    main(args.input, args.table, args.n_threads)