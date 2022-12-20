""" Component that takes an image of a bookshelf and segments it into books"""

import cv2
from os import path

class SegmentBooks:
    
    def __init__(self, book_segmenter):
        self.book_segmenter = book_segmenter

    def segment_books(self, parent_id, book_shelf_image, output_directory):
        load_uri = path.join(output_directory, book_shelf_image)
        image_cv2 = cv2.imread(load_uri)

        book_images, book_regions = self.book_segmenter.segmentBooks(image_cv2)

        books = []
        
        for index, book_image in enumerate(book_images):
            #if index == 0:
            #    print ("book_image",book_image)
            file = parent_id+"_book_"+str(index)
            filename = file+".jpg"
            output_uri = path.join(output_directory, filename)
            success = cv2.imwrite(output_uri, book_image)
            if(success):
                books.append({"filename":filename, "book_id":file})
            else:
                print("problem with book segmentation", output_uri, index)
        return books
