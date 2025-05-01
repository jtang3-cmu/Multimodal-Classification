# run_multithreaded_batch.py
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from skimage.io import imsave
from tqdm import tqdm
from oct_edge_detector import OCTEdgeDetector

def process_image(path_tuple):
    path_str, flat_out_dir, skel_out_dir, overwrite = path_tuple
    detector = OCTEdgeDetector()
    img_path = Path(path_str)
    flat_path = Path(flat_out_dir) / f"{img_path.stem}_flat.png"
    skel_path = Path(skel_out_dir) / f"{img_path.stem}_skeleton.png"

    if not overwrite and flat_path.exists() and skel_path.exists():
        return f"[SKIP] {img_path.name}"

    try:
        results = detector.run(str(img_path))
        flat_img = (results['flattened'] * 255).astype('uint8')
        skel_img = (results['skeleton'].astype('uint8')) * 255
        imsave(str(flat_path), flat_img)
        imsave(str(skel_path), skel_img)
        return f"[OK] {img_path.name}"
    except Exception as e:
        return f"[ERROR] {img_path.name}: {e}"

def batch_run_parallel(input_folder, output_flat_dir, output_skel_dir, overwrite=False, extensions={'.jpg', '.png', '.bmp'}):
    Path(output_flat_dir).mkdir(parents=True, exist_ok=True)
    Path(output_skel_dir).mkdir(parents=True, exist_ok=True)

    image_paths = [
        str(p) for p in Path(input_folder).rglob('*')
        if p.suffix.lower() in extensions
    ]

    args = [(p, output_flat_dir, output_skel_dir, overwrite) for p in image_paths]

    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_image, args), total=len(args), desc="Processing images"):
            print(result)

if __name__ == "__main__":
    batch_run_parallel(
        input_folder="/Users/martingoessweiner/CMU Biophotonics Lab Dropbox/CMU Biophotonics/Users/Martin/Courses/Spring 2025/AI in BME/Project/Data/Cirrus",
        output_flat_dir="/Users/martingoessweiner/CMU Biophotonics Lab Dropbox/CMU Biophotonics/Users/Martin/Courses/Spring 2025/AI in BME/Project/Data/Cirrus/flattened",
        output_skel_dir="/Users/martingoessweiner/CMU Biophotonics Lab Dropbox/CMU Biophotonics/Users/Martin/Courses/Spring 2025/AI in BME/Project/Data/Cirrus/skeleton",
        overwrite=False
    )
