import numpy as np


def validateBook(cols, rows, region, imShape):
    """
    Converted from Matlab

    Uses width and diagonal heuristic to validate whether a connected component feature is a valid book.

    :param cols: A list of x coordinate in this book region
    :param rows: A list of y coordinate in this book region
    :param region: The four corner coordinates of the book
    :param imShape: The (height, width) size of the original image.
    :return: True if this area is a valid book.
    """
    height, width = imShape

    minY = np.min(cols)
    maxY = np.max(cols)
    minX = np.min(rows)
    maxX = np.max(rows)

    # Figure out which region coordinate is which
    # This assumes that the average y coordinate splits the top and bottom of the quadrilateral
    centerY = np.sum(region[:, 1]) / 4.0
    topPoints = region[region[:, 1] < centerY]
    btmPoints = region[region[:, 1] > centerY]
    topCenterX = np.sum(topPoints[:, 0]) / 2.0
    btmCenterX = np.sum(btmPoints[:, 0]) / 2.0

    # If the average y coordinate does not split the shape in 2, discard
    if len(topPoints) != 2 or len(btmPoints) != 2:
        return False

    topLeft = topPoints[topPoints[:, 0] < topCenterX]
    topRight = topPoints[topPoints[:, 0] > topCenterX]
    btmLeft = btmPoints[btmPoints[:, 0] < btmCenterX]
    btmRight = btmPoints[btmPoints[:, 0] > btmCenterX]

    if len(topLeft) != 1 or len(topRight) != 1 or len(btmLeft) != 1 or len(btmRight) != 1:
        return False

    topLeft = topLeft[0]
    topRight = topRight[0]
    btmLeft = btmLeft[0]
    btmRight = btmRight[0]

    topWidth = abs(topRight[0] - topLeft[0])
    btmWidth = abs(btmRight[0] - btmLeft[0])
    avgWidth = max(topWidth, btmWidth)
    widthDiffPrct = max(btmWidth, topWidth) / max(min(btmWidth, topWidth), .1)

    heightThreshold = 0.7
    widthThresholdLower = 0.01
    widthThresholdUpper = 0.7
    widthDiffThreshold = 3
    # ratio between width and height of a book
    widthHeightRatioThresholdUpper = 0.25
    widthHeightRatioThresholdLower = 0.02

    bookWidth = abs(maxX - minX)
    bookHeight = abs(maxY - minY)
    bookDiagonal = np.sqrt((maxY - minY) ** 2 + (maxX - minX) ** 2)

    heightOK = bookDiagonal >= height * heightThreshold
    widthOK = width * widthThresholdLower <= bookWidth <= height * widthThresholdUpper
    widthOK = widthOK and widthDiffPrct < widthDiffThreshold
    widthHeight = widthHeightRatioThresholdLower < (avgWidth / bookHeight) < widthHeightRatioThresholdUpper

    return heightOK and widthOK and widthHeight
