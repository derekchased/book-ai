import PIL.Image
import cv2
import numpy as np
from skimage import morphology

from segmentation.matlab.bookOrient import bookOrient
from segmentation.matlab.searchLine import findLinesFast
from segmentation.matlab.validateBook import validateBook


def findBookLines(bookShelf):
    """
    Converted from Matlab

    Finds lines in an image of a plank of books.

    :param bookShelf: A color image of a book plank.
    :return: The lines found
    """
    bookSpineImage = cv2.cvtColor(bookShelf, cv2.COLOR_BGR2GRAY)
    height, width = bookShelf.shape[:2]

    matlabThreshold = 0.1
    upperThreshold = matlabThreshold * 255
    lowerThreshold = upperThreshold * 0.4

    # Matlab canny is different from opencv canny. This smoothing should help to make them more similar
    # See also: https://dsp.stackexchange.com/questions/4716/differences-between-opencv-canny-and-matlab-canny
    smoothedInput = cv2.GaussianBlur(bookSpineImage, (7, 7), sigmaX=2)
    binary = cv2.Canny(smoothedInput, lowerThreshold, upperThreshold)

    threshold = 0.3

    startPix = int(np.ceil(-width / 2))
    endPix = int(np.ceil(width / 2))

    lines = findLinesFast(binary, threshold, startPix, endPix)

    return lines


def CCA(lineImg, inputImage):
    """
    Does connected component analysis to find connected areas in the `finalDetectLine` image.
    Components are passed through validateBook() to see if they are valid book candidates after which the books are
    extracted from the original color image `inputImage`.

    :param lineImg: A BW image containing all lines
    :param inputImage: The original RGB input image.
    :return: An array of book images.
    """

    lineImg = cv2.bitwise_not(lineImg)

    numComponents, componentImg = cv2.connectedComponents(lineImg, connectivity=8)

    # Arrays to store results
    books = []
    regions = []

    # Component 0 is the lines. skip over that
    for componentNum in range(1, numComponents):
        imCopy = np.copy(inputImage)

        cols, rows = np.transpose(np.argwhere(componentImg == componentNum))

        minCol = np.min(cols)
        maxCol = np.max(cols)
        minRow = np.min(rows)
        maxRow = np.max(rows)

        if minCol == maxCol or minRow == maxRow:
            continue

        bookMask = componentImg[minCol:maxCol, minRow:maxRow] == componentNum

        contours, _ = cv2.findContours(bookMask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        region = simplify_contour(contours[0], 4)

        # Reject shapes that are not easily made into a rectangle
        if region is None or len(region) != 4:
            continue

        if validateBook(cols, rows, region, lineImg.shape[:2]):
            extractedBook = imCopy[minCol:maxCol, minRow:maxRow, :]

            deg = bookOrient(bookMask)

            region[:, 0] += minRow
            region[:, 1] += minCol

            regions.append(region)

            # Set non-book areas to black
            extractedBook[bookMask == 0, :] = [0, 0, 0]

            # Rotate book image
            pil_image = PIL.Image.fromarray(extractedBook)
            rotated = pil_image.rotate(deg, resample=PIL.Image.BILINEAR)
            extractedBook = np.array(rotated)

            books.append(crop_image(extractedBook))

    if len(regions) > 0:
        # Sort books by leftmost x coordinate
        order = np.argsort(np.min(np.array(regions)[:, :, 0], axis=1))
        # Numpy doesn't support ragged arrays so use list comprehension instead of books[order]
        # List comprehension requires the 'inverse' of argsort, which is argsort applied twice
        order = np.argsort(order)
        books = [books[i] for i in order]
        regions = [regions[i] for i in order]

    return books, regions

def simplify_contour(contour, n_corners=4):
    """
    Binary searches best `epsilon` value to force contour
        approximation contain exactly `n_corners` points.

    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.

    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    """
    n_iter, max_iter = 0, 100
    lb, ub = 0., .5

    while True:
        n_iter += 1
        if n_iter > max_iter:
            return None

        k = (lb + ub)/2.
        eps = k * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) > n_corners:
            lb = (lb + ub) / 2.
        elif len(approx) < n_corners:
            ub = (lb + ub) / 2.
        else:
            return np.squeeze(approx)

def crop_image(img):
    """
    Crops out black edges on the outside of an image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    m, n = img.shape[:2]
    mask0, mask1 = thresh.any(0), thresh.any(1)
    col_start, col_end = mask0.argmax(), n-mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m-mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]
