import os
import re
from glob import glob

import cv2

img2 = cv2.imread(f'./outputPlot.jpg')
img1 = img2.copy()
img2.fill(255)

MAX_TEXT_WIDTH = img2.shape[1] - 100
title_list = []

# find last shelltesting output folder
subfolders = glob("./pipeline_test_output/*/")
subfolders = sorted(subfolders)
folder = subfolders[-1]


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

offset = 80
x, y = 100, 100

cnt = 0
for [idx, lbl] in title_list:
    if lbl == "No results found":
        continue

    text = "Title " + str(idx)+ ": " + str(lbl)
    textlen_orig = len(text)

    # truncate text
    textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 2, 3)[0]

    while textsize[0] + x > MAX_TEXT_WIDTH:
        text = text[:-1]
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 2, 3)[0]

    if textlen_orig != len(text):
        text += '...'

    cv2.putText(img2, text, (x,y+offset*cnt), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,0), 3)
    cnt += 1

h_img = cv2.hconcat([img1, img2])

cv2.imwrite(f'./outputPlotWithText.jpg', h_img)
