import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit
from skimage import morphology, measure, util
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from segmentation.matlab.findBookLines import findBookLines, CCA


class BookSegmenter:

    def __init__(self):
        self.__debug_mode = False
        self.__debug_folder = None
        self.__debug_img_cnt = 0

    def setDebugFolder(self, debug_folder):
        """
        Set the debug folder. If a folder is passed here, debug output images will be written to this folder.
        This folder is created if it does not exist.

        If `None` is passed, no debug images will be created. Writing debug images slows down the program significantly.

        :param debug_folder: A string containing the debug folder.
        """
        self.__debug_mode = debug_folder is not None
        self.__debug_folder = debug_folder
        self.__debug_img_cnt = 0

    def segmentToBooks(self, image):
        """
        Segments an image of a bookshelf into images of books.
        :param image: An image file, read using the `cv2.imread()` function
        :return: A 3-tuple, containing an array of images for each book,
                 the region (4 xy coordinates, one for each corner) each plank was extracted from, and
                 the region that each book was extracted from.
        """
        shelves, shelfRegions = self.segmentBookshelves(image)

        allBooks = []
        bookRegions = []

        for idx in range(len(shelves)):
            shelf = shelves[idx]
            shelfRegion = shelfRegions[idx]

            books, regions = self.segmentBooks(shelf)

            # Find the inverse transform to convert book coordinates in
            # the plank reference frame back to original image reference frame
            h, _, _ = self.__findTransform(shelfRegion, inverse=True)

            for regionIdx in range(len(regions)):
                region = regions[regionIdx]
                region = np.expand_dims(region, 1)
                region = region.astype(np.float32)
                transformedRegion = cv2.perspectiveTransform(region, h)

                regions[regionIdx] = np.squeeze(transformedRegion)

            allBooks.append(books)
            bookRegions.append(regions)

        return allBooks, shelfRegions, bookRegions

    def segmentBookshelves(self, image):
        """
        Segments an image of a bookshelf into different planks.
        The regions containing planks are returned as an array containing the four corners for every region that
        contains a plank.
        """
        # Processing is done on a downscaled version of the image.
        scaleFactor = 0.5
        resized = self.__resizeImage(image, scaleFactor)

        height, width = resized.shape[:2]

        voteThreshold = 150
        minLineLength = width * 0.2
        maxLineGap = width * 0.05
        angleMargin = 20

        self.__debug_write_image(image, 'original')
        self.__debug_write_image(resized, 'resized')

        binary = self.__imageToBinary(resized, 0.5)
        filtered = self.__filterFeatures(binary, 0.9)
        dilated = self.__dilateImage(filtered)
        lines = self.__findLinesInRange(dilated, 0, angleMargin, voteThreshold, minLineLength, maxLineGap)
        self.__visualizeLines(lines, resized.shape[:2], 'houghlines')
        lines = self.__clusterLines(lines, resized.shape[:2])
        regions = self.__extractRegions(lines, resized.shape[:2])
        regions = self.__filterPlankRegions(regions)

        if self.__debug_mode:
            print(f'Segmented into {len(regions)} shelves!')

        # Upscale region to fit original image.
        regions = [self.__scaleRegion(region, 1.0 / scaleFactor) for region in regions]

        images = []

        for region in regions:
            shelf = self.__extractRegion(image, region)
            self.__debug_write_image(shelf, 'shelf')
            images.append(shelf)

        return images, np.array(regions)

    def segmentBooks(self, image):
        """
        Segment book spines.
        Returns the regions containing book spines as an array containing the four corners for every region in the image
        that contains a book spine.
        """
        height = 700
        scaleFactor = height / image.shape[0]
        resized = self.__resizeImage(image, scaleFactor)

        lines = findBookLines(resized)

        transposedShape = (resized.shape[1], resized.shape[0])
        self.__visualizeLines(lines, transposedShape, 'lines')

        lines = self.__clusterLines(lines, transposedShape, 60, False, 1)

        # Upscale lines
        lines[:, 1] = lines[:, 1] * (1/scaleFactor)

        lineImg = self.__drawLines(lines, (image.shape[1], image.shape[0]))

        lineImg = np.transpose(lineImg)
        self.__debug_write_image(lineImg, 'lines')

        books, regions = CCA(lineImg, image)

        if self.__debug_mode:
            print(f'Segmented plank into {len(books)} books!')

        if self.__debug_mode:
            for book in books:
                self.__debug_write_image(book, 'book')

        return books, np.array(regions)

    def __resizeImage(self, image, scaleFactor):
        """
        Resize image width and height by scaleFactor
        """
        height = int(image.shape[0] * scaleFactor)
        width = int(image.shape[1] * scaleFactor)
        dim = (width, height)

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def __imageToBinary(self, image, threshold):
        """
        Converts image to binary by applying a Gaussian blur followed by Canny edge detection.
        """
        upperThreshold = threshold * 255
        lowerThreshold = upperThreshold * 0.4

        # Matlab canny is different from opencv canny. This smoothing should help to make them more similar
        # See also: https://dsp.stackexchange.com/questions/4716/differences-between-opencv-canny-and-matlab-canny
        smoothed = cv2.GaussianBlur(image, (7, 7), sigmaX=2)
        binary = cv2.Canny(smoothed, lowerThreshold, upperThreshold)

        self.__debug_write_image(binary, 'binary')

        return binary

    def __filterFeatures(self, binary, eccentricityPercentile):
        """
        Filter features in B&W image by eccentricity.
        """
        labeled = measure.label(binary > 0, connectivity=2)  # ensure input is binary
        data = measure.regionprops_table(labeled, properties=('label', 'eccentricity'))

        table = pd.DataFrame(data)
        threshold = table.eccentricity.quantile(eccentricityPercentile)

        eccentric_labels = table['label'] * (table['eccentricity'] > threshold)
        new_labels = util.map_array(
            labeled,
            np.asarray(table['label']),
            np.asarray(eccentric_labels),
        )

        filtered = ((new_labels > 0) * 255).astype(np.uint8)
        self.__debug_write_image(filtered, 'filtered1')

        filtered = morphology.remove_small_objects(filtered > 0, 50, connectivity=2)

        filtered = (filtered * 255).astype(np.uint8)
        self.__debug_write_image(filtered, 'filtered2')

        return filtered

    def __dilateImage(self, binary):
        """
        Dilates binary image
        """
        strel = morphology.rectangle(3, 1)
        dilated = morphology.binary_dilation(binary, selem=strel).astype(np.uint8)

        dilated *= 255

        self.__debug_write_image(dilated, 'dilated')

        return dilated

    def __findLinesInRange(self, image, angle, angleMargin, voteThreshold, minLineLength, maxLineGap, rho=3):
        """
        Uses the Hough Line Transform to find lines in a B&W image that are within a certain angle range.
        Lines shorter than minLineLength are not considered.
        """
        # Find lines in image
        lines = cv2.HoughLinesP(image, rho=rho, theta=np.pi / 180, threshold=voteThreshold, minLineLength=minLineLength,
                                maxLineGap=maxLineGap)

        # An array containing polynomial coefficients for each line found. (i.e. `a` and `b` in `y = a*x + b`).
        result = []

        if lines is None:
            return []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            theta = np.arctan2((y2 - y1), (x2 - x1))
            deg = theta * 180 / np.pi

            if self.__isAngleBetween(angle - angleMargin, angle + angleMargin, deg):
                (b, a) = polyfit([x1, x2], [y1, y2], 1)
                result.append((a, b))

        return result

    def __drawLines(self, lines, imShape):
        height, width = imShape

        mask = np.zeros(imShape, dtype="uint8")

        for line in lines:
            (slope, intercept) = line
            x1, y1 = (0, intercept)
            x2, y2 = (width, intercept + width * slope)

            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)

        return mask

    def __visualizeLines(self, lines, imShape, name):
        if not self.__debug_mode:
            return

        mask = self.__drawLines(lines, imShape)
        self.__debug_write_image(mask, name)

    def __isAngleBetween(self, start, end, angle):
        """
        Simple function to check whether an angle lies between start and end.
        Based on:
        https://math.stackexchange.com/questions/1044905/simple-angle-between-two-angles-of-circle/3316065#3316065
        """
        end = end - start + 360 if (end - start) < 0 else end - start
        angle = angle - start + 360 if (angle - start) < 0 else angle - start
        return angle < end

    def __clusterLines(self, lines, imShape, maxK=11, ignoreSlope=True, threshold=0.5):
        """
        Clusters a set of lines using the k-means transform on their polynomial coefficients. This clustering only
        looks at the intercept position, since the slopes are already constrained in the `__findLinesInRange` function.

        :returns the means of the line clusters found.
        """
        if len(lines) == 0:
            return np.array([])

        scaler = StandardScaler()
        scaledLines = scaler.fit_transform(lines)

        if ignoreSlope:
            # Since the line angles are already with a desired range, we set them all to 0 here.
            # This improves clustering and allows for the mean angle to be extracted by calling inverse_transform.
            scaledLines[:, 0] = scaledLines[:, 0] * 0
        else:
            scaledLines[:, 1] = scaledLines[:, 1] * 10

        kmeansArgs = {
            "init": "k-means++",
            "n_init": 15,
            "max_iter": 300,
            # "random_state": 42,
            # "algorithm": "full"
        }

        kRange = range(1, min(maxK, len(lines) + 1))

        bestMeans = []
        k = 0
        for k in kRange:
            kmeans = KMeans(n_clusters=k, **kmeansArgs)
            kmeans.fit(scaledLines)
            if kmeans.inertia_ < threshold:
                bestMeans = kmeans.cluster_centers_
                break
            elif k == kRange[-1]:
                bestMeans = kmeans.cluster_centers_

        if not self.__debug_mode:
            if not ignoreSlope:
                bestMeans[:, 1] = bestMeans[:, 1] / 10
            return scaler.inverse_transform(bestMeans)

        print(f'Number of clusters: {k}')

        # Debug plot and image
        meansX, meansY = np.transpose(np.rot90(bestMeans, 2))
        slopes, intercepts = np.transpose(scaledLines)

        plt.plot(intercepts, slopes, 'bx')
        plt.plot(meansX, meansY, 'go')
        plt.xlabel("intercepts")
        plt.ylabel("slopes")

        # write plot to file
        io_buf = io.BytesIO()
        plt.gcf().savefig(io_buf, format='png', dpi=500)
        io_buf.seek(0)
        img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
        io_buf.close()
        img = cv2.imdecode(img_arr, 1)
        plt.clf()

        self.__debug_write_image(img, 'clusters')

        if not ignoreSlope:
            bestMeans[:, 1] = bestMeans[:, 1] / 10

        self.__visualizeLines(scaler.inverse_transform(bestMeans), imShape, 'meanlines')

        return scaler.inverse_transform(bestMeans)

    def __extractRegions(self, lines, imShape):
        """
        Converts lines in an image to regions in between the lines.
        This function assumes lines don't cross.
        """
        height, width = imShape

        # Sort lines by intersect position
        if len(lines) > 0:
            lines = lines[np.argsort(lines[:, 1])]

        # Add the top and bottom edge of the image as a line demarcating a region.
        imageTop = np.array([[0, 0]])
        imageBottom = np.array([[0, height]])
        if len(lines) > 0:
            lines = np.concatenate([imageTop, lines, imageBottom])
        else:
            lines = np.concatenate([imageTop, imageBottom])

        # Construct regions of four coordinates.
        regions = []

        for idx in range(1, len(lines)):
            topLine = lines[idx - 1]
            btmLine = lines[idx]

            topLeft = topLine[1]
            bottomLeft = btmLine[1]
            topRight = topLeft + topLine[0] * width
            bottomRight = bottomLeft + btmLine[0] * width

            if bottomRight < 0:
                bottomRight = 0
            if topRight < 0:
                topRight = 0
            if topRight > height:
                topRight = height
            if bottomRight > height:
                bottomRight = height

            corners = np.array([
                [0, topLeft],
                [width, topRight],
                [width, bottomRight],
                [0, bottomLeft]
            ])

            regions.append(corners)

        return np.array(regions)

    def __filterPlankRegions(self, regions):
        """
        Filters regions that are too small based on the assumption that bookshelves should have similar height.
        """

        # Find the height of each region.
        heights = []
        for region in regions:
            topLeft = region[0][1]
            bottomLeft = region[3][1]

            height = bottomLeft - topLeft
            heights.append(height)

        heights = np.array(heights)

        # Filter planks with a height of less than 100 pixels, since those are usually regions
        # between the top and bottom edge of the same plank
        filteredHeights = heights[heights > 100]

        # Find median height of the remaining regions
        medianHeight = np.median(filteredHeights)

        # Shelves should at least be as tall as 85% of the height of the median shelf.
        minHeight = medianHeight * 0.85

        filteredRegions = []

        for region in regions:
            topLeft = region[0][1]
            bottomLeft = region[3][1]

            height = bottomLeft - topLeft

            if minHeight <= height:
                filteredRegions.append(region)

        return np.array(filteredRegions)

    def __scaleRegion(self, sourcePts, scaleFactor):
        """
        Scales a region by scaleFactor
        """
        return (sourcePts * scaleFactor).astype(int)

    def __findTransform(self, sourcePts, inverse=False):
        (minX, minY) = np.amin(sourcePts, axis=0)
        (maxX, maxY) = np.amax(sourcePts, axis=0)

        width = maxX - minX
        height = maxY - minY

        # Output image size.
        destPts = np.array([
            [0.0, 0.0],
            [float(width), 0.0],
            [float(width), float(height)],
            [0.0, float(height)]
        ])

        if inverse:
            tmp = sourcePts
            sourcePts = destPts
            destPts = tmp

        h, _ = cv2.findHomography(sourcePts, destPts)
        return h, width, height

    def __extractRegion(self, image, sourcePts):
        """
        Warps a quadrilateral region from an image, defined by a list of four corner coordinates, onto the smallest
        enclosing rectangle.
        :param sourcePts: An array of four integer (x,y) coordinates defining the corners
                          of the quadrilateral in the image.
        """
        h, width, height = self.__findTransform(sourcePts)
        warped = cv2.warpPerspective(image, h, (width, height))

        return warped

    def __cropShape(self, image, contour):
        """
        Crops a shape from an image.
        :param image:
        :param contour:
        :return:
        """
        (minX, minY) = np.amin(contour, axis=0)
        (maxX, maxY) = np.amax(contour, axis=0)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)

        mask = cv2.bitwise_not(mask)
        r, c = np.where(mask)
        result = image.copy()
        result[r, c] = (0, 0, 0)
        result = result[minY:maxY, minX:maxX]
        return result

    def __debug_write_image(self, image, name):
        """
        Helper function to write a debug image to the debug folder.
        """
        if self.__debug_mode:
            cv2.imwrite(f'{self.__debug_folder}/{self.__debug_img_cnt}-{name}.jpg', image)
            self.__debug_img_cnt += 1
