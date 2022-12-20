import numba as nb

@nb.njit(cache=True)
def concat(arrays):
    """
    Simple numba version of np.concatenate
    :param arrays: Array of arrays
    :return: The flattened input
    """
    count = len(arrays)
    result = []

    for idx in range(count):
        result.extend(arrays[idx])

    return result
