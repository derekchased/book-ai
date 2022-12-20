""" Component that accepts an image file and copies it into the project directory """

import shutil
import os 
from os import path

class SubmitPhoto:
    
    # def __init__(self):
    #     pass

    def submit(self, original_uri, new_filename, output_directory):
        output_uri = os.path.join(output_directory, new_filename)
        return shutil.copy(original_uri, output_uri)

# if __name__ == "__main__":
#     submit_photo = SubmitPhoto("testing")
#     print(submit_photo.submit("./submittedphotos/IMG_0745.jpg","IMG_0745_TIMESTAMP9842.jpg"))
#     #print(os.path.splitext("hello.tiff")[0])
