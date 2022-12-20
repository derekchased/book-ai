import numba as nb
import numpy as np

from segmentation.numba.numba_concat import concat
from segmentation.numba.numba_countnonzero import countNonzeroAt
from segmentation.numba.numba_polyfit import fit_poly


@nb.njit(parallel=True, cache=True)
def searchLine(binary, startY, thresholdVal):
    """
    Converted from matlab (heavily optimized)

    Tries to detect lines in the `binary` image.

    NOTE: Returns lines in a transposed coordinate system, since a vertical line
          cannot be described with the y=ax+b notation

    :param binary:       Some 2d binary image as a numpy array.
    :param startY:       An integer y coordinate.
    :param thresholdVal: A floating point threshold value. If a larger proportion of the pixels under a line
                         scan is 1 than this proportion value, a line is detected.

    :return: An array of lines in the transposed coordinate system.
    """

    height, width = binary.shape[:2]

    y1 = startY

    centerLine = np.ceil(binary.shape[0] / 2)
    x2 = centerLine

    m = -y1 / x2

    noneLine = (np.Inf, np.Inf)

    lines = [noneLine] * width

    for x1 in nb.prange(width):
        y1 = startY + x1

        startRow = 0
        endRow = height

        if m > 0:
            startRow = (-y1 / m) + x1
            endRow = (width - y1) / m + x1
        elif m < 0:
            startRow = ((width - 1) - y1) / m + x1
            endRow = (-y1 / m) + x1

        startRow = max(int(np.ceil(startRow)), 0)
        endRow = min(int(np.floor(endRow)), height)

        rows = np.arange(startRow, endRow)
        cols = np.floor((y1 + m * (rows - x1))).astype(np.int64)

        count = countNonzeroAt(binary, rows, cols)
        total = rows.shape[0]

        if count > total * thresholdVal:
            startCol = cols[0]
            endCol = cols[-1]

            xs = np.array([startRow, endRow]).astype(np.float64)
            ys = np.array([startCol, endCol]).astype(np.float64)

            coeffs = fit_poly(xs, ys, deg=1)

            slope = coeffs[0]
            intercept = coeffs[1]

            if abs(slope) < 1:
                lines[x1] = (slope, intercept)

    lines = [line for line in lines if line != noneLine]

    return lines


@nb.njit(parallel=True, cache=True)
def findLinesFast(binary, threshold, startPix, endPix):
    """
    Finds lines in binary image (see searchLine for parameters)
    :return: List of lines found as 2-tuples (slope, intercept) for y = slope * x + intercept
    """

    # Pre-initialize array to allow parallel execution of for loop.
    rangeLen = endPix - startPix
    noneLine = [(np.Inf, np.Inf)]
    lines = [noneLine] * rangeLen

    for theInc in nb.prange(startPix, endPix):
        returnedLines = searchLine(binary, theInc, threshold)

        lines[theInc] = returnedLines

    lines = [line for line in lines if len(line) > 0 and not (line == noneLine)]

    return concat(lines)


@nb.njit(parallel=True, cache=True)
def findAnglesFast(binary, threshold, startPix, endPix):
    """
    Finds the angle of lines in binary image (see searchLine for parameters)
    :return List of pseudo-angle values, one for each line found
    """
    # Numba doesn't support iteration in steps greater than 1, so we have to fix that.
    startPix = startPix // 3
    endPix = endPix // 3

    # Pre-initialize array to allow parallel execution of for loop.
    rangeLen = abs(endPix - startPix)
    noneVal = np.Inf
    incVals = [noneVal] * rangeLen

    for idx in nb.prange(startPix, endPix):
        theInc = int(idx * 3)

        lines = searchLine(binary, theInc, threshold)

        # If a line is found, add this angle to the list
        if len(lines) > 0:
            incVals[idx] = theInc

    incVals = [val for val in incVals if val != noneVal]

    return incVals
