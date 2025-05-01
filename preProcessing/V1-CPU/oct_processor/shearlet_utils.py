import numpy as np
from pyshearlab import SLgetShearletSystem2D, SLsheardec2D, SLnormalizeCoefficients2D, modulate2, dfilters
from scipy.signal import hilbert

def get_shearlet_coeffs(image, nScales, shearLevels):
    shearLevels = np.array(shearLevels) 
    nx, ny = image.shape
    # directionalFilter = modulate2(dfilters('cd', 'd')[0], 'r')
    shearletSystem = SLgetShearletSystem2D(1, nx, ny, nScales, shearLevels)
    coeffs = SLsheardec2D(image, shearletSystem)
    normalized = SLnormalizeCoefficients2D(coeffs, {
        'RMS': shearletSystem['RMS'],
        'nShearlets': shearletSystem['shearlets'].shape[2]
    })

    if not np.iscomplexobj(normalized):
        imaginary = np.empty_like(normalized)
        for i in range(normalized.shape[2]):
            analytic_signal = hilbert(normalized[:, :, i], axis=0)
            imaginary[:, :, i] = np.imag(analytic_signal)
        even = normalized
        odd = imaginary
    else:
        even = np.real(normalized)
        odd = np.imag(normalized)

    return even, odd