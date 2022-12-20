import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
from skimage.transform import rotate
import numpy as np
import CRAFT.craft_utils
import CRAFT.test as test
import CRAFT.imgproc as imgproc
import CRAFT.file_utils as file_utils

import json
import zipfile
import pandas as pd
from CRAFT.craft import CRAFT

from collections import OrderedDict

#from google.colab.patches import cv2_imshow


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

enable_cuda = torch.cuda.is_available()
canvas_size = 1000
#CRAFT
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='CRAFT/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=enable_cuda, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=canvas_size, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='test-data', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='CRAFT/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
# Note: make sure not to use a refiner model as this file isn't in the repo currently...

args = parser.parse_args()


#SET DEGREE AMOUNT AND NUMBER OF ROTATIONS
degree = 90
num_rot = 4


def set_degree(d):
    degree = d


def set_num_rot(n):
    num_rot = n


def update_directory(directory):
    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(directory)

    image_names = []

    start = directory

    for num in range(len(image_list)):
        image_names.append(os.path.relpath(image_list[num], start))

    return image_list, image_names, start


def main(directory):
    """
    Main entry point to the CRAFT module.
    Takes all segmented book spines, produces their 4 rotations (0°, 90°, 180°, 270°) and does localisation on those.
    Resulting images are saved to subfolders of the provided directory folder.

    Args:
        directory: the path of the input folder (containing segmented book spine images)
    """
    image_list, image_names, start = update_directory(directory)
    result_folder = directory + "/results"
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    before_pandas = time.time()
    # Make a new dataframe to later be exported to .csv
    data=pd.DataFrame(columns=['image_name', 'rotation', 'word_bboxes', 'pred_words', 'align_text'])
    image_names_quadruple = [x for x in image_names for _ in range(4)]
    data['image_name'] = image_names_quadruple
    print("Elapsed time for pandas dataframe instantiation: " + str(time.time() - before_pandas))

    # load net
    pre_instantiate_CRAFT = time.time()
    net = CRAFT()     # initialize
    print("Elapsed time to instantiate CRAFT: "+str(time.time()-pre_instantiate_CRAFT))

    pre_load = time.time()
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(test.copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(test.copyStateDict(torch.load(args.trained_model, map_location=(None if enable_cuda else torch.device('cpu')))))

    print("Elapsed time for loading weights: "+str(time.time()-pre_load))

    more_stuff = time.time()
    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location=(None if enable_cuda else torch.device('cpu')))))

        refine_net.eval()
        args.poly = True
    print("Elapsed time for rest: "+str(time.time()-more_stuff))

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        #print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path))
        images = []

        file_name, file_ext = os.path.splitext(os.path.basename(image_path))

        image = imgproc.loadImage(image_path)
        file_path = directory+'/rotated/'
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
        for i in range(num_rot):
            images.append(rotate(image, i*degree, resize=True, preserve_range=True))
            image_path = file_path + file_name + '_' + str(i * degree) + file_ext
            #print(os.path.basename(image_path))
            data['image_name'][k * num_rot + i] = os.path.basename(image_path)
            io.imsave(image_path, images[i].astype(np.uint8))
            #cv2.imwrite(image_path, images[i])
            img = imgproc.loadImage(image_path)
            # We might be able to skip the entire saving thing by passing images[i].astype instead of img
            # But this is for demoing
            bboxes, polys, score_text, det_scores = test.test_net(net, img, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args, refine_net)

            #print("bboxes: {}".format(bboxes))

            bbox_score={}

            for box_num in range(len(bboxes)):
              key = str (det_scores[box_num])
              item = bboxes[box_num]
              bbox_score[key]=item

            data['word_bboxes'][k*num_rot+i]=bbox_score
            data['rotation'][k * num_rot + i] = i*degree
            #data['word_bboxes'][k]=bbox_score
            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))

            if not os.path.isdir(result_folder + '/masks/'):
                os.mkdir(result_folder + '/masks/')

            mask_file = result_folder + '/masks/res_' + filename + '_' + str(i*degree) + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)

            # This line saves the images with overlayed text bounding boxes ('polys')
            #print("Image path: {}".format(image_path))

            # We might need a better folder management
            file_utils.saveResult(image_path, img[:,:,::-1], polys, dirname=(result_folder+'/localised/'))

    data.to_csv(directory+'/results/data.csv', sep = ',', na_rep='Unknown')
    print("elapsed time : {}s".format(time.time() - t))


if __name__ == '__main__':
    main()