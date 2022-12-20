""" Component that accepts book spine images and performs localization """


from os import path
import cv2
from skimage import io
import CRAFT.test as test
import CRAFT.imgproc as imgproc
import CRAFT.file_utils as file_utils
from skimage.transform import rotate

class TextLocalization:
    
    def __init__(self, net, args, refine_net):
        self.net = net
        self.args = args
        self.refine_net = refine_net
        self.degree = 90
        self.num_rot = 4
        

    def localize_text(self, parent_id, book_spine_image, output_directory):
        # Load the original image of the book spine
        load_uri = path.join(output_directory, book_spine_image)
        book_spine_image = imgproc.loadImage(load_uri)

        # Create empty list to store the different localizations based on the image rotations
        localized = []

        # Iterate over each rotation
        for i in range(self.num_rot):

            current_rotation = i*self.degree
            
            # ID of this rotation
            base_filename = parent_id + "_rot_" + str(current_rotation)

            # First rotate the image and save it
            # skimage.transform.rotate
            rotated_book_spine_image = rotate(book_spine_image, current_rotation, resize=True, preserve_range=True)
            rotated_book_spine_image_file_name = base_filename + "_rotated.jpg"
            rotated_book_spine_image_uri = path.join(output_directory, rotated_book_spine_image_file_name)
            
            # Question for sonja: Is this the same as cv2? Can we use one or the other instead of both?
            io.imsave(rotated_book_spine_image_uri, rotated_book_spine_image) 

            # Load the image using craft
            rotated_book_spine_image = imgproc.loadImage(rotated_book_spine_image_uri) # TODO can this load the numpy data which is already in memory?
            
            # Process with craft net
            # Sonja: We might be able to skip the entire saving thing by passing images[i].astype instead of img
            # Sonja: But this is for demoing
            bboxes, polys, score_text, det_scores = test.test_net(self.net, rotated_book_spine_image, self.args.text_threshold, self.args.link_threshold, self.args.low_text, self.args.cuda, self.args.poly, self.args, self.refine_net)
            bbox_score={}
            for box_num in range(len(bboxes)):
              key = str (det_scores[box_num])
              item = bboxes[box_num]
              bbox_score[key]=item

            # Create masked image
            rotated_book_spine_image_mask_file_name = base_filename + "_mask.jpg"
            rotated_book_spine_image_mask_uri = path.join(output_directory, rotated_book_spine_image_mask_file_name)
            cv2.imwrite(rotated_book_spine_image_mask_uri, score_text)

            # Create localized image (with bounding boxes plotted)
            rotated_book_spine_image_localized_file_name = base_filename + "_localized.jpg"
            file_utils.saveResult(rotated_book_spine_image_localized_file_name, rotated_book_spine_image[:,:,::-1], polys, dirname=output_directory+"/")

            # Add a data object of this rotation to the list
            localized.append({
                "rotation_file_name":rotated_book_spine_image_file_name,
                "localized_file_name":rotated_book_spine_image_localized_file_name,
                "mask_file_name":rotated_book_spine_image_mask_file_name,
                "rotation":str(current_rotation),
                "localization_id":base_filename
                })

        return localized