import os
import numpy as np
import cv2
import pandas as pd
#from google.colab.patches import cv2_imshow


def crop(pts, image):
    """
    Takes inputs as 8 points and returns cropped, masked image with a white background

    Args:
        pts: list of 8 corner points (x and y for each of the 4 corners)
        image: the input image containing the corner points

    Returns:
        a cropped image of the text specified by the input coordinates
    """
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    cropped = image[y:y + h, x:x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2


def generate_words(image_name, score_bbox, image, directory):
    """
    Processes coordinate information from the output .csv from the CRAFT module, crops text boxes and saves the cropped images

    Args:
        image_name: the name of the segmented book spine image
        score_bbox: list of text box coordinates as written in the .csv output from the CRAFT module
        image: book spine image
        directory: path for the output of cropped images (cropped images are saved to directory/cropped/)
    """
    num_bboxes = len(score_bbox)
    for num in range(num_bboxes):
        bbox_coords = score_bbox[num].split(':')[-1].split(',\n')
        if bbox_coords != ['{}']:
            l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0])
            t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1])
            r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0])
            t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1])
            r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
            b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
            l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
            b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']'))
            pts = np.array([[int(l_t), int(t_l)], [int(r_t), int(t_r)], [int(r_b), int(b_r)], [int(l_b), int(b_l)]])

            if np.all(pts) > 0:
                width = np.linalg.norm(np.subtract([int(r_t), int(t_r)], [int(l_t), int(t_l)]))
                height = np.linalg.norm(np.subtract([int(l_t), int(t_l)], [int(l_b), int(b_l)]))
                #print("file: {}, num: {}, width: {}, height: {}".format(image_name, num ,width, height))
                if(height > width):
                    continue
                word = crop(pts, image)

                folder = '/'.join(image_name.split('/')[:-1])

                # CHANGE DIR
                dir = directory+'/cropped/'

                if os.path.isdir(os.path.join(dir + folder)) == False:
                    os.makedirs(os.path.join(dir + folder))

                try:
                    x, y, c = image.shape
                    file_name = os.path.join(dir + image_name)
                    corners = [round(x, 3) for x in [l_t, t_l, r_t, t_r, r_b, b_r, l_b, b_l]]
                    cv2.imwrite(
                        file_name + '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(corners[0], corners[1], corners[2], corners[3], corners[4], corners[5], corners[6], corners[7], y, x), word)
                    #print('Image saved to ' + file_name + '_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(l_t, t_l, r_t, t_r, r_b,
                    #                                                                            b_r, l_b, b_l))
                except:
                    continue


# Note that this path only works locally
def perform_crop(directory):
    """
    Performs cropping for all book spines contained in the output .csv of the CRAFT module

    Args:
        directory: path to 'localizationocr' or where the segmented book spines are located
    """
    data = pd.read_csv(directory+'/results/data.csv')
    start = directory+'/rotated/'
    for image_num in range(data.shape[0]):
        #name = data['image_name'][image_num] + '_' + str(data['rotation'][image_num])
        path = os.path.join(start, data['image_name'][image_num])
        #print(path)
        image = cv2.imread(path)
        image_name = data['image_name'][image_num].strip('.jpg')
        #image_name is just the name of the image in the csv, like, "Encyclopedia-of..._270"
        score_bbox = data['word_bboxes'][image_num].split('),')
        #print(score_bbox)
        generate_words(image_name, score_bbox, image, directory)