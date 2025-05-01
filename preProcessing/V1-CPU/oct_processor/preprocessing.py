from PIL import Image
import numpy as np
from skimage.transform import resize
import scipy.ndimage as ndi
from bm3d import bm3d, BM3DProfile

def load_and_prepare_image(file_path, qLen, device, denoise_sigma):
    image = Image.open(file_path).convert('L')

    if device == 'Heidelberg':
        image = image.crop((1540, 500, 2304, 1010))

    resized_img = image.resize((qLen, qLen))
    data = np.asarray(resized_img, dtype=np.float32) / 255.0
    data = ndi.median_filter(data, size=(1, 3))

    img_min, img_max = np.min(data), np.max(data)
    normalized = (data - img_min) / (img_max - img_min)
    denoised = bm3d(normalized, denoise_sigma, profile=BM3DProfile())
    denoised_resized = resize(denoised, (qLen, qLen), preserve_range=True, anti_aliasing=True)

    return denoised_resized

