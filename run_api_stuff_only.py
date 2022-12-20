from searchstring_and_API import retrieve_results_from_API
from searchstring_and_API import retrieve_books
import sys

if __name__ == '__main__':
    directory = 'pipeline_test_output/shelltesting0521153143'#substitute with the location of the csv file obtained from the OCR part
    if(len(sys.argv) > 1):
        directory = str(sys.argv)[1]
    #retrieve_results_from_API(directory+"/localizationocr/results/data.csv", directory, directory + "/localizationocr/")
    out = retrieve_books(directory + "/localizationocr/results/data.csv", directory, directory + "/localizationocr/")
    # print(out)
    # out.to_csv(directory+"/localizationocr/results/books.csv")