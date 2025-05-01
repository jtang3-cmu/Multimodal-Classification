import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from cuShearLab import (
    SLgetShearletSystem2D,
    SLsheardec2D,
    SLnormalizeCoefficients2D
)
# ------------------------------
# Utility functions
# ------------------------------
def get_gaussian_kernel(size=5, sigma=1.0):
    coords = torch.arange(size) - size // 2
    grid = coords[:, None] ** 2 + coords[None, :] ** 2
    kernel = torch.exp(-grid / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.view(1, 1, size, size)

def apply_gaussian_blur(img: torch.Tensor, kernel: torch.Tensor):
    img = img.unsqueeze(0)  # [1, 1, H, W]
    return F.conv2d(img, kernel, padding=kernel.shape[-1] // 2).squeeze(0)

# ------------------------------
# Config
# ------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Running on Metal")
else:
    device = torch.device("cpu")
    print("Running on CPU")


image_path = '/Users/martingoessweiner/CMU Biophotonics Lab Dropbox/CMU Biophotonics/Users/Martin/Courses/Spring 2025/AI in BME/Project/Data/Cirrus/B-Scans/OrgImg_0001.jpg'
qLen = 512 # need 224x224
nScales = 2
shearLevels = [1, 2]  # match PyShearLab

# ------------------------------
# Image Preprocessing
# ------------------------------
image = Image.open(image_path).convert('L')
image_tensor = TF.to_tensor(image).to(device)  # [1, H, W]
image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(qLen, qLen), mode='bilinear', align_corners=False).squeeze(0)

# Normalize
min_val, max_val = image_tensor.min(), image_tensor.max()
norm_tensor = (image_tensor - min_val) / (max_val - min_val + 1e-8)

# Optional denoise with blur
kernel = get_gaussian_kernel(size=5, sigma=1.0).to(device)
norm_tensor = apply_gaussian_blur(norm_tensor, kernel)
norm_tensor = norm_tensor.squeeze(0)
# ------------------------------
# Build Shearlet System & Transform
# ------------------------------
# Build Shearlet System
shearlet_system = SLgetShearletSystem2D(
    useGPU=(device.type == 'cuda' or device.type == 'mps'),
    rows=qLen,
    cols=qLen,
    nScales=nScales,
    shearLevels=torch.tensor(shearLevels),
    full=False,
    device=device
)

# # Shearlet Transform
coeffs = SLsheardec2D(norm_tensor, shearlet_system)

# # Normalize
coeffs_norm = SLnormalizeCoefficients2D(coeffs, shearlet_system)

# # ------------------------------
# # Normalize & Edge Feature Extraction
# # ------------------------------
reconScales = [0]


even = coeffs_norm.real
odd = coeffs_norm.imag

H, W, N = coeffs.shape
nOrient = N // nScales
preferred_even = torch.zeros((H, W, len(reconScales)), device=device)
preferred_odd = torch.zeros((H, W, len(reconScales)), device=device)



for k, j in enumerate(reconScales):
    start = j * nOrient
    end = (j + 1) * nOrient
    even_j = even[:, :, start:end]
    odd_j = odd[:, :, start:end]

    idx = torch.argmax(torch.abs(odd_j), dim=2)
    I = torch.arange(H).view(-1, 1).expand(H, W)
    J = torch.arange(W).view(1, -1).expand(H, W)

    preferred_even[:, :, k] = even_j[I, J, idx]
    preferred_odd[:, :, k] = odd_j[I, J, idx]

sum_odd = torch.sum(preferred_odd, dim=2)
sum_even = torch.sum(preferred_even, dim=2)
max_odd = torch.max(preferred_odd, dim=2).values

T = 10e-2
epsilon = -1e-10
threshold = 0.2

# T = 1e20
# epsilon = 0.3 # For Cirrus
# epsilon = 0.15 # For Heidelberg

edge_measure = (sum_odd - sum_even - nScales * T) / (nScales * max_odd + epsilon)
# edge_measure = torch.clamp(edge_measure, 0, 1)
binary_edges = edge_measure > threshold

# # # ------------------------------
# # # Visualization
# # # ------------------------------
image_np = norm_tensor.cpu().numpy()
rgb = np.stack([image_np] * 3, axis=-1)
overlay = rgb.copy()
overlay[binary_edges.cpu().numpy()] = [1, 0, 0]  # Red

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image_np, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(edge_measure.cpu().numpy(), cmap='plasma')
axes[1].set_title('Edge Measure')
axes[1].axis('off')
fig.colorbar(axes[1].images[0], ax=axes[1], shrink=0.8)

axes[2].imshow(overlay)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()
