""" Component that takes an image of a bookcase and segments it into bookshelves"""

import cv2
from os import path

class SegmentBookcase:
    
    def __init__(self, book_segmenter):
        self.book_segmenter = book_segmenter

    def segment_bookcase(self, parent_id, book_case_image, output_directory):
        load_uri = path.join(output_directory, book_case_image)
        image_cv2 = cv2.imread(load_uri)

        shelf_images, shelf_regions = self.book_segmenter.segmentBookshelves(image_cv2)

        shelves = []
        
        for index, shelf_image in enumerate(shelf_images):
            file = parent_id+"_shelf_"+str(index)
            filename = file + ".jpg"
            output_uri = path.join(output_directory, filename)
            success = cv2.imwrite(output_uri, shelf_image)
            if(success):
                shelves.append({"filename":filename, "bookshelf_id":file})
            else:
                print("problem with segmentation", output_uri, index)
        return shelves
        
        