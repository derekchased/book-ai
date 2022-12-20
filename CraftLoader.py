""" Wrapper for loading craft """

import argparse
import CRAFT.file_utils as file_utils
import os
from CRAFT.craft import CRAFT
import CRAFT.test as test
import torch


class CraftLoader:
    
    def __init__(self):
        print("CraftLoader::init")

        self.cuda_enabled = torch.cuda.is_available()

        #CRAFT
        parser = argparse.ArgumentParser(description='CRAFT Text Detection')
        parser.add_argument('--trained_model', default='CRAFT/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
        parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
        parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
        parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
        parser.add_argument('--cuda', default=self.cuda_enabled, type=str2bool, help='Use cuda for inference')
        parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
        parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
        parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
        parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
        parser.add_argument('--test_folder', default='test-data', type=str, help='folder path to input images')
        parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
        parser.add_argument('--refiner_model', default='CRAFT/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
        # Note: make sure not to use a refiner model as this file isn't in the repo currently...

        self.args = parser.parse_args()


        """ For test images in a folder """
        self.image_list, _, _ = file_utils.get_files(self.args.test_folder)

        self.image_names = []
        image_paths = []

        #CUSTOMISE START
        start = self.args.test_folder

        #SET self.degree AMOUNT AND NUMBER OF ROTATIONS
        self.degree = 90
        self.num_rot = 4

        for num in range(len(self.image_list)):
            self.image_names.append(os.path.relpath(self.image_list[num], start))

        # load self.net
        self.net = CRAFT()     # initialize

        print('Loading weights from checkpoint (' + self.args.trained_model + ')')
        if self.args.cuda:
            self.net.load_state_dict(test.copyStateDict(torch.load(self.args.trained_model, map_location=(None if self.cuda_enabled else torch.device('cpu')))))
        else:
            self.net.load_state_dict(test.copyStateDict(torch.load(self.args.trained_model, map_location=(None if self.cuda_enabled else torch.device('cpu')))))

        if self.args.cuda:
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False

        self.net.eval()

        # LinkRefiner
        self.refine_net = None
        if self.args.refine:
            from refinenet import RefineNet
            self.refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.args.refiner_model + ')')
            if self.args.cuda:
                self.refine_net.load_state_dict(copyStateDict(torch.load(self.args.refiner_model)))
                self.refine_net = self.refine_net.cuda()
                self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(copyStateDict(torch.load(self.args.refiner_model, map_location=(None if self.cuda_enabled else torch.device('cpu')))))

            self.refine_net.eval()
            self.args.poly = True
    
    def get_net(self):
        return self.net

    def get_refine_net(self):
        return self.refine_net

    def get_args(self):
        return self.args

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

if __name__ == '__main__':
    tl = CraftLoader()
    # tl.localize_text("test_id")