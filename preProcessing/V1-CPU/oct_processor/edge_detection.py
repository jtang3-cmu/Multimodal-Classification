import numpy as np

def extract_edges(even_coeffs, odd_coeffs, nScales, reconScales, epsilon, T):
    nOrient = even_coeffs.shape[2] // nScales
    nx, ny, _ = even_coeffs.shape

    preferred_odd = np.zeros((nx, ny, len(reconScales)))
    preferred_even = np.zeros((nx, ny, len(reconScales)))

    for k, j in enumerate(reconScales):
        start, end = j * nOrient, (j + 1) * nOrient
        odd_j, even_j = odd_coeffs[:, :, start:end], even_coeffs[:, :, start:end]
        idx = np.argmax(np.abs(odd_j), axis=2)
        I, J = np.arange(nx)[:, None], np.arange(ny)[None, :]
        preferred_odd[:, :, k] = odd_j[I, J, idx]
        preferred_even[:, :, k] = even_j[I, J, idx]

    sum_odd = np.sum(preferred_odd, axis=2)
    sum_even = np.sum(preferred_even, axis=2)
    max_odd = np.max(preferred_odd, axis=2)

    edge_measure = (sum_odd - sum_even - (nScales * T)) / (nScales * max_odd + epsilon)
    return np.clip(edge_measure, 0, 1)

