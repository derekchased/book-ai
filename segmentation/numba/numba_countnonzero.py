import numba as nb


@nb.njit(cache=True)
def countNonzeroAt(array, rows, cols):
    """
    Simple numba version of np.count_nonzero
    :param array: Input array
    :param rows: Row indices
    :param cols: Column indices
    :return: The amount of nonzero elements
    """
    count = 0

    for i in range(len(rows)):
        if array[rows[i], cols[i]] != 0:
            count += 1
    return count