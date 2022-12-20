""" Pipeline class, processes the pipeline from start to finish """
import sys

from LocalizationOCR import LocalizationOCR
from searchstring_and_API import retrieve_results_from_API
from searchstring_and_API import retrieve_books
from os import path
import shutil
import cv2
import pandas as pd
import numpy as np
import os
import re
from glob import glob

class BookAIPipeline:

    def __init__(self, book_ai_data, process, submit_photo, book_segmenter,
        segment_bookcase=None, segment_books=None, craft_loader=None, text_localization=None, text_recognition=None,
        text_assembly=None, book_match=None):

        # init objects
        self.book_ai_data = book_ai_data
        self.process = process
        self.submit_photo = submit_photo
        self.book_segmenter = book_segmenter
        self.segment_bookcase = segment_bookcase
        self.segment_books = segment_books
        self.text_localization = text_localization
        self.text_recognition = text_recognition
        self.text_assembly = text_assembly
        self.book_match = book_match
        self.craft_loader = craft_loader

        # Store directory used often
        self.directory = process["directory"]

        # Input image directory
        self.input_directory = process["original_uri"]

        # start pipeline
        self.main()

        # print and write pipeline results
        print("")
        print(self.book_ai_data.get_json(self.process))
        self.book_ai_data.write_json(self.process)

    def main(self):
        self.do_submit()

    def do_submit(self):

        print("=Submit Photo=")

        original_uri = self.process["original_uri"]
        file_name = self.process["file_name"]
        photo_submit_success = self.submit_photo.submit(original_uri, file_name, self.directory)

        if photo_submit_success:
            self.do_segmentation_localization_ocr()
        else:
            print("could not create image")
            sys.exit()

    def plottingOutput(self, shelfRegions, bookRegions):
        print("PLOTTING OUTPUT")
        img = cv2.imread(self.input_directory)

        # read the columns with shelf number and book number in the csv_file
        csv_file = self.directory+"/localizationocr/results/data.csv"
        cols_list = ["shelf", "book"]
        csv_data =  pd.read_csv(csv_file, usecols=cols_list)
        shelves = csv_data["shelf"].values.tolist()
        books = csv_data["book"].values.tolist()

        uniq, indices = np.unique(shelves, return_index=True)

        text_shelf = "shelf-"
        text_book = "book-"

        # plot the book regions
        j = 0
        k = 0
        region_number = 0
        indmin = 0
        indmax = 0
        foundList = []
        foundIndex = []
        foundBooks = []
        # regionArr for the planks
        for regionArr in bookRegions:
            i = 0
            indmin = indmax
            if region_number + 1 < len(indices):
                indmax = int(indices[region_number+1])
            else:
                indmax = len(shelves) - 1
            # region for each book (or not book) in the shelf
            for region in regionArr:
                foundBooks.append(region)
                new_text_book = text_book + str(i)
                # does the book exist in the correct shelf in the .csv file
                for l in range(indmin,indmax):
                    item = books[l]
                    if item == new_text_book:
                        j += 1
                        foundList.append(k)
                        foundIndex.append(j)
                        break
                i += 1
                k += 1
            region_number += 1

        # White mask for non-book areas
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # plot the book boundaries
        for i in foundList:
            region = foundBooks[i]
            pts = np.array(region, dtype=np.int32)
            # -1 means numpy will figure out the dimension
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 255), 3)
            cv2.fillPoly(mask, [pts], 255)

        # plot the planks
        for region in shelfRegions:
            pts = np.array(region, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 0, 0), 100)

        # plot the text
        for m in range(len(foundList)):
            idx = foundList[m]
            region = foundBooks[idx]
            # print book number on image
            font = cv2.FONT_HERSHEY_COMPLEX
            # compute average bottom coordinate of book region
            centerY = np.sum(region[:, 1]) / 4.0
            btmPoints = region[region[:, 1] > centerY]
            org = np.sum(btmPoints, axis=0) // 2
            fontScale = 2
            # white color in BGR
            color2 = (255, 255, 255)
            # line thickness of 2 px
            thickness = 8
            text = str(foundIndex[m])

            # get text size to center horizontally
            textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]

            textX = int(org[0] - (textsize[0] / 2))  # center horizontally
            textY = int(org[1] + (textsize[1] / 2))  # move down vertically

            img = cv2.putText(img, text, (textX, textY), font, fontScale, color2, thickness, cv2.LINE_AA)

        # Draw 'non-book area' mask
        mask = cv2.bitwise_not(mask)
        # Erode so we don't draw over lines
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        img = cv2.addWeighted(img, 1, mask, 0.5, 0)

        #cv2.imwrite(f'./outputPlot.jpg', img)
        cv2.imwrite(self.directory+'/outputPlot.jpg', img)
        img2 = img.copy()
        img2.fill(255)

        MAX_TEXT_WIDTH = img2.shape[1] - 100
        title_list = []

        # find last shelltesting output folder
        #subfolders = glob("./pipeline_test_output/*/")
        #subfolders = sorted(subfolders)
        #folder = subfolders[-1]
        folder = self.directory


        with open(f"{folder}/Recognized_books.txt", "r") as f:
            lines = f.readlines()

            bookNumber = 0
            line = 0
            bookText = ''
            for item in lines:

                # Book number is on line 1 (second line)
                if line == 1:
                    match = re.search(r'Matched with segmented book #([0-9]+)', item)

                    if not match:
                        raise Exception("Cannot match book with segmented. Are you using an old recognized_books.txt?")

                    bookNumber = int(match.group(1))

                # Book title is on line 3
                if line == 3:
                    #item = item[:70]
                    if item.find("No results found for") != -1:
                        bookText = "No results found"
                    else:
                        bookText = item[:-1] # last character is newline

                # author name is on line 4
                if line == 4:
                    authors = []
                    # Match everything between single quotes
                    for author_match in re.finditer(r"'(.*?)'", item):
                        authors.append(author_match.group(1))

                    if len(authors) > 0:
                        bookText += f' - {", ".join(authors)}'

                line += 1

                if item.find('---') != -1:
                    title_list.append([bookNumber, bookText])
                    bookText = ''
                    line = 0

        f.close()

        #list = ['Title 1: ', 'Title 2:', 'Title 3:', 'Title 4:', 'Title 5:', 'Title 6:']

        offset = 60 #80
        x, y = 100, 100
        MAX_HEIGHT = img2.shape[0]-offset

        cnt = 0
        newindex = 1
        for [idx, lbl] in title_list:
            if lbl == "No results found":
                newindex += 1
                continue

            text = "Book " + str(newindex)+ ": " + str(lbl) #str(idx)
            textlen_orig = len(text)

            # truncate text
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 2, 3)[0]

            while textsize[0] + x > MAX_TEXT_WIDTH:
                text = text[:-1]
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 2, 3)[0]

            if textlen_orig != len(text):
                text += '...'

            if (y+offset*cnt) > MAX_HEIGHT:
                cv2.putText(img2, "...", (x,MAX_HEIGHT), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2)
                break

            cv2.putText(img2, text, (x,y+offset*cnt), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 2) # last 3
            cnt += 1
            newindex += 1

        h_img = cv2.hconcat([img, img2])

        #cv2.imwrite(f'./outputPlotWithInfo.jpg', h_img)
        cv2.imwrite(self.directory+'/outputPlotWithInfo.jpg', h_img)



    def do_segmentation_localization_ocr(self):
        print("=Segment Photo=")
        root_process_id = self.process["process_id"]
        file_name = self.process["file_name"]

        load_uri = path.join(self.directory, file_name)
        image_cv2 = cv2.imread(load_uri)

        # Segment the book case image and return list of filenames
        # bookshelves = self.segment_bookcase.segment_bookcase(root_process_id, file_name, self.directory)
        book_shelves, shelfRegions, bookRegions = self.book_segmenter.segmentToBooks(image_cv2)

        for shelf_index, book_shelf in enumerate(book_shelves):
            for book_index, book_image in enumerate(book_shelf):
                #if index == 0:
                #    print ("book_image",book_image)
                # file = parent_id+"_book_"+str(index)
                file = root_process_id+"_shelf_"+str(shelf_index)+"_book_"+str(book_index)
                filename = file+".jpg"
                output_uri = path.join(self.directory+"/localizationocr", filename)
                success = cv2.imwrite(output_uri, book_image)

                # if(success):
                #     books.append({"filename":filename, "book_id":file})
                # else:
                #     print("problem with book segmentation", output_uri, index)
        print("=Localization and OCR=")
        self.locr = LocalizationOCR(self.directory+"/localizationocr")

        self.do_book_search()

        self.plottingOutput(shelfRegions, bookRegions)

    def do_book_search(self):
        print("=Book Search=")
        # The first line uses Ludwig's original implementation
        # The second one uses the rotation dependent search string assembly (including everything else Ludwig did if the string is still bad)
        #retrieve_results_from_API(self.directory+"/localizationocr/results/data.csv", self.directory, self.directory + "/localizationocr/")
        out = retrieve_books(self.directory + "/localizationocr/results/data.csv", self.directory, self.directory + "/localizationocr/")
        # out is a pandas dataframe that includes one row per book with the information from GoogleBooksAPI
        # print(out)
        # out.to_csv(self.directory + "/localizationocr/results/books.csv")
