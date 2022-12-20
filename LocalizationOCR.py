""" Component that accepts book spine images and performs localization """

import os.path
import shutil
import time

import CRAFT.pipeline as craft
import CRAFT.crop_images as crop
import OCR.recognition as ocr
import group_boxes as group



class LocalizationOCR:
    
    def __init__(self, directory):
        is_test = False
        
        # DELETE OLD FOLDERS
        if os.path.isdir('./results'):
            shutil.rmtree('./results')
        if os.path.isdir('./rotated'):
            shutil.rmtree('./rotated')
        if os.path.isdir('./cropped'):
            shutil.rmtree('./cropped')

        # DELETE LOG FILE
        if os.path.exists('./OCR/log_demo_result.txt'):
            os.remove('./OCR/log_demo_result.txt')

        # CRAFT
        before = time.time()
        craft.set_degree(90)
        craft.set_num_rot(4)
        craft.main(directory)
        after_craft = time.time()
        print('\t--- ELAPSED TIME FOR CRAFT ---')
        print('\t\t{}'.format(after_craft-before))

        # CROPPING
        crop.perform_crop(directory)
        after_crop = time.time()
        print('\t--- ELAPSED TIME FOR CROP ---')
        print('\t\t{}'.format(after_crop-after_craft))

        # OCR
        ocr.main(directory)
        after_ocr = time.time()
        print('\t--- ELAPSED TIME FOR OCR ---')
        print('\t\t{}'.format(after_ocr-after_crop))

        # GROUPING
        group.main(directory)
        after = time.time()
        print('\t--- ELAPSED TIME FOR GROUPING ---')
        print('\t\t{}'.format(after-after_ocr))
        total = after - before
        print("TOTAL ELAPSED TIME FOR LOCALISATION AND OCR: {}".format(total))

        if is_test:

            print("Degree: {}, Rotations: {}".format(90, 4))
            for i in range(4):
                print("\tIteration: {}".format(i))
                craft.set_degree(90)
                craft.set_num_rot(4)
                craft.main()

            print()
            print("Degree: {}, Rotations: {}".format(90, 2))
            for i in range(4):
                print("\tIteration: {}".format(i))
                craft.set_degree(90)
                craft.set_num_rot(2)
                craft.main()