import cv2
import numpy as np
from skimage import morphology

from segmentation.matlab.searchLine import findAnglesFast


def bookOrient(bookSegment):
    """
    This tries to determine the orientation angle of a book spine.

    :param bookSegment: A BW image with book area as nonzero.
    :return: The orientation of the book in degrees.
    """
    scaleFactor = 0.1
    width = int(bookSegment.shape[1] * scaleFactor)
    height = int(bookSegment.shape[0] * scaleFactor)

    theTemp = cv2.resize(bookSegment.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)

    # Morphological thinning reduces the binary matrix to a pixel-wide image.
    thinImage = morphology.thin(theTemp)

    thresholdVal = 0.35

    thinImage[thinImage > 0] = 255
    thinImage = thinImage.astype(np.uint8)

    # These are the bounds for angle rotation
    startPix = int(np.ceil(-width))
    endPix = int(np.ceil(width))

    incVal = findAnglesFast(thinImage, thresholdVal, startPix, endPix)

    deg = calculateAngle(incVal, width, height)
    deg = -1 * deg

    return deg


def calculateAngle(incr, width, height):
    """
    Converted from Matlab

    Calculates the average angle for a set of lines using an array of pseudo-angle values `incr`.

    :param incr: A list of y coordinates
    :param width: The width of the image
    :param height: The height of the image
    :return: An angle in degrees
    """
    if len(incr) == 0:
        return 0

    avgInc = np.ceil(np.mean(incr))
    oppositeVal = np.ceil(width / 2) + avgInc
    tanValue = ((width / 2 - oppositeVal) / (height / 2))
    deg = np.arctan(tanValue) / np.pi * 180

    return deg
