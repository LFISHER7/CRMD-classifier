import os
import numpy as np
from PIL import Image

def get_image_paths(base_folder_path):
    """Gets the paths of all files downstream of folder"""
    image_paths = []
    for root, dirs, files in os.walk(base_folder_path, topdown=False):
        for file in files:  # files are file names
            paths = os.path.join(root, file)  # joins file name with root to get path for each file
            image_paths.append(paths)
    image_paths = ([f for f in image_paths if not f.endswith('.DS_Store')])
    return image_paths

def mean_image(folder_path):
    """Take the mean image of all images within folder path"""
    image_paths = get_image_paths(folder_path)
    width, height = Image.open(image_paths[0]).size  #Get dimension of first image
    number_of_files = len(image_paths)
    arr = np.zeros((height, width, 3), np.float) #Create numpy array to store average
    for image in image_paths:
        imarr = np.array(Image.open(image), dtype=np.float)
        arr = arr+imarr/number_of_files
    arr = np.array(np.round(arr), dtype=np.uint8) #Round values and cast as 8-bit (256)
    out = Image.fromarray(arr, mode="RGB") #Generating image
    new_image_directory = folder_path.split('/')
    new_image_directory[-2] = new_image_directory[-2] + '-average'
    new_image_directory = '/'.join(new_image_directory)
    print(new_image_directory)
    out.save(new_image_directory, format='png')

def produce_mean_image(folder_path):
    """Gets all the file paths within folder, takes mean of images in each folder"""
    file_paths = []
    for root, dirs, files in os.walk(folder_path, topdown=False):
        file_paths.append(root)
    file_paths = file_paths[:-1]
    for file in file_paths:
        mean_image(file)

# produce_mean_image('../labels/train')
# produce_mean_image('../labels/test')  