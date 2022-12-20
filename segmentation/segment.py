import glob
import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt

from segmentation.BookSegmenter import BookSegmenter

if __name__ == '__main__':
    debugMode = True

    segmenter = BookSegmenter()

    for idx in range(15):
        img = cv2.imread(f'./images/im{idx}.jpg')

        if debugMode:
            folder = f'./images/debug/{idx}'

            # Create folder for debug images or delete its contents.
            if not os.path.isdir(folder):
                os.mkdir(folder)
            else:
                files = glob.glob(f'{folder}/*')
                for f in files:
                    os.remove(f)
        else:
            folder = None

        startTime = time.time()

        segmenter.setDebugFolder(folder)
        books, shelfRegions, bookRegions = segmenter.segmentToBooks(img)

        print(f'Segmenting image ./images/im{idx}.jpg took {time.time() - startTime} seconds')

        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Draw book outlines
        # Use nice colors for borders
        cmap = plt.get_cmap("jet")

        for regionArr in bookRegions:
            N = len(regionArr)
            cnt = 0
            for region in regionArr:
                pts = np.array(region, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))
                color = np.array(cmap(cnt / N)) * 255
                cv2.polylines(img, [pts], True, color, 3)
                cv2.fillPoly(mask, [pts], 255)
                cnt += 1

        # Draw shelves
        for region in shelfRegions:
            pts = np.array(region, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (255, 0, 0), 5)

        # Draw 'non-book area' mask
        mask = cv2.bitwise_not(mask)
        # Erode so we don't draw over lines
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        img = cv2.addWeighted(img, 1, mask, 0.5, 0)

        # Write result
        cv2.imwrite(f'./images/im{idx}-result.jpg', img)
