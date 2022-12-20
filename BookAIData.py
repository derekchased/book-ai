""" Data interfact class, creates a base "process", writes data from each 
component to the process, prints to json, writes to json, converts json to csv  """
import time
import os
from os import path
import json 

class BookAIData:
    
    def __init__(self):
        pass

    def create_new_process(self, directory, file_name, output_directory):
        if not path.isdir(output_directory):
            os.makedirs(output_directory)
        if not path.isdir(output_directory+"/localizationocr"):
            os.makedirs(output_directory+"/localizationocr")
        original_uri = os.path.join(directory, file_name)
        file, extension = os.path.splitext(file_name)
        process_id = self.generate_id(file)
        new_file_name = process_id + extension
        return {"process_id":process_id, "original_uri":original_uri, "file_name":new_file_name, "directory":output_directory}

    def generate_id(self, file):
        return file + time.strftime("%y%m%d%H%M%S")
    
    ##############################
    # Components
    ##############################

    def book_case_segmentation_success(self, root_process, filename, bookshelf_id):
        if "bookshelf" not in root_process:
            root_process["bookshelf"] = []
        bookshelves = root_process["bookshelf"]
        bookshelves.append({"file_name":filename, "bookshelf_id":bookshelf_id})
        return bookshelves

    def book_segmentation_success(self, parent_process, filename, book_id):
        if "book" not in parent_process:
            parent_process["book"] = []
        books = parent_process["book"]
        books.append({"file_name":filename, "book_id":book_id})
        return books

    def text_localization_success(self, parent_process, localized_file_name, mask_file_name,
                            rotation, localization_id):#, word_bboxes):
        if "localization" not in parent_process:
            parent_process["localization"] = []
        localizations = parent_process["localization"]
        localizations.append({"localized_file_name":localized_file_name, 
            "mask_file_name":mask_file_name, "rotation":rotation, 
            "localization_id":localization_id})#, "word_bboxes":word_bboxes })
        return localizations


    ##############################
    # Output from program
    ##############################    

    # def transform_json_csv(self, process, indent=2):
    #     print(json.dumps(process,indent=indent))
    #     pass

    # def write_csv(self):
    #     pass

    def get_json(self, process, indent=2):
        return json.dumps(process,indent=indent)

    def write_json(self, process, indent=2):
        # the json file where the output must be stored
        output_uri = os.path.join(process["directory"], process["process_id"]+".json")
        out_file = open(output_uri, "w")
        json.dump(process, out_file, indent = indent)
        out_file.close()


        