import os
import numpy
from PIL import Image
from mean_image import get_image_paths

def std_image(folder_path):
    """Take the mean image of all images within folder path"""
    image_paths = get_image_paths(folder_path)
    width, height = Image.open(image_paths[0]).size  #Get dimension of first image
    number_of_files = len(image_paths)
    arr = numpy.zeros((height, width, 3), numpy.float) #Create numpy array to store average
    zero_arr = numpy.zeros((height, width, 3), numpy.float)
    for image in image_paths:
        print(image)
        imarr = numpy.array(Image.open(image), dtype=numpy.float)
        arr = arr+imarr/number_of_files
        diff_arr = numpy.subtract(imarr, arr)
        square_arr = numpy.square(diff_arr)
        zero_arr += square_arr/number_of_files
    final_arr = numpy.sqrt(zero_arr) #standard deviation of image as array
    arr = numpy.array(numpy.round(final_arr), dtype=numpy.uint8) #Round values, cast as 8-bit(256)
    out = Image.fromarray(arr, mode="RGB") #Generating image
    new_image_directory = folder_path.split('/')
    new_image_directory[-2] = new_image_directory[-2] + '-std'
    new_image_directory = '/'.join(new_image_directory)
    print(new_image_directory)
    out.save(new_image_directory, format='png')


def produce_std_image(folder_path):
    """Gets all the file paths within folder, takes std of images in each folder"""
    file_paths = []
    for root, dirs, files in os.walk(folder_path, topdown=False):
        file_paths.append(root)
    file_paths = file_paths[:-1]
    for file in file_paths:
        std_image(file)

# produce_std_image('../labels/train')
