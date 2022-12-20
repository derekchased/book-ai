""" Factory class to handle instantiation of objects """

from BookAIData import BookAIData
from BookAIPipeline import BookAIPipeline
from SubmitPhoto import SubmitPhoto
from segmentation.BookSegmenter import BookSegmenter
from SegmentBookcase import SegmentBookcase
from SegmentBooks import SegmentBooks
from CraftLoader import CraftLoader
from TextLocalization import TextLocalization

class ComponentFactory:
    
    def __init__(self):
        pass

    def get_component(self, args):
        return getattr(self,args[0])(args) # args[1:]

    def bookai(self, args):
        # Parse args
        original_directory = args[1]
        original_file_name = args[2]
        output_directory = args[3]

        # Database representation
        bookai_data = BookAIData()

        # Initialize new process
        process = bookai_data.create_new_process(original_directory, original_file_name, output_directory)

        # Create Components
        submit_photo = SubmitPhoto()
        book_segmenter = BookSegmenter()
        # segment_bookcase = SegmentBookcase(book_segmenter)
        # segment_books = SegmentBooks(book_segmenter)
        # craft_loader = CraftLoader()
        # text_localization = TextLocalization(craft_loader.get_net(), craft_loader.get_args(), craft_loader.get_refine_net())
        

        # Create Pipeline
        # bookai = BookAIPipeline(bookai_data, process, submit_photo, segment_bookcase, segment_books, craft_loader, text_localization)
        bookai = BookAIPipeline(bookai_data, process, submit_photo, book_segmenter)#, segment_bookcase, segment_books)

        return bookai