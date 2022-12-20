""" Shell Program to Run Book AI Program, should interact with the
command line and instantiate what needs to be instantiated. Currently
instantiating through the componentfactory class """

import time # to be removed
import sys
from ComponentFactory import ComponentFactory

class BookAIShell:

    def __init__(self, args):
        self.component_factory = ComponentFactory()

        # Setup app environment
        # self.__create_venv()
        # self.__install_dependencies()
        # self.__load_modules()
        self.__process_args(args)




    def __process_args(self, args):
        self.component = self.component_factory.get_component(args)

        #print("BShell__process_args",self.component)

        # for index, arg in enumerate(args):
        #     if arg == "bookai":
        #         continue
        #     elif arg == "segment":
        #         continue
        #     elif arg == "segment":
        #         continue
        #     else:
        #         continue



    def __segment(self, args):
        """Takes a path to an image of a bookshelf and
        segments that image into individual books. Images
        for each shelf are stored and images for each book
        are stored

        Args:
            args (array): command line arguments
            example: bookai segment IMG_422.jpg

        Returns:
            success (str): id of this process or an error code
            the id is used by other modules to locate images and log
            or error log
            example: img_422_042921_100454
        """
        print(args)
        return success

    def __localize(self, args):
        """Takes a path to an image of a bookshelf and
        segments that image into individual books. Images
        for each shelf are stored and images for each book
        are stored

        Args:
            args (array): command line arguments
            example: bookai segment IMG_422.jpg

        Returns:
            success (str): id of this process or an error code
            the id is used by other modules to locate images and log
            or error log
            example: img_422_042921_100454
        """
        print(args)
        return success




if __name__ == "__main__":

    #args = ["bookai", ".", "IMG_0745.jpg","pipeline_test_output/shelltesting"+time.strftime("%m%d%H%M%S")]
    # Demo image:
    args = ["bookai", "./segmentation/images/", "im6.jpg", "pipeline_test_output/shelltesting" + time.strftime("%m%d%H%M%S")]

    before = time.time()

    shell = BookAIShell(args)

    elapsed_time = "TOTAL ELAPSED TIME: {}".format(time.time()-before)
    line = "-"*len(elapsed_time)+"-"*22
    spaces = " "*10
    print(line)
    print("|"+spaces+elapsed_time+spaces+"|")
    print(line)
