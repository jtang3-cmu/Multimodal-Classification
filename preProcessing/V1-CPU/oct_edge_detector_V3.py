from oct_processor import process_oct_image

# Need some device logic
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
        T1 = 1e-20
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
        epsilon2 = 0.15
        footprint = [3, 30]

    return (denoise_sigma, nScales1, shearLevels1, reconScales1, T1, epsilon1, flatten_mode, poly_deg, padding_top, padding_bot, flatten_sigma, nScales2, shearLevels2, reconScales2, T2, epsilon2, footprint)

        
device = "Cirrus"

(denoise_sigma, nScales1, shearLevels1, reconScales1, T1, epsilon1, flatten_mode, poly_deg, padding_top, padding_bot, flatten_sigma, nScales2, shearLevels2, reconScales2, T2, epsilon2, footprint) = get_device_params(device)


# Run Preprocessing
skeleton, edge_map, flat_image = process_oct_image(
    file_path="Data/Cirrus/B-Scans/RID_1001000095_20180701_15_L_OPT_512x1024x128_Original_ORG_IMG_JPG_0094.jpg",
    device="Cirrus",
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
    edge_threshold=0.2,
    visualize=True
)