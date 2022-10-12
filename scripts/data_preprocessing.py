import shutil
import numpy as np
from skimage import exposure, io
from mean_image import get_image_paths

def copy_folder_tree(folder_path, new_folder_name):
    """Copies folder, and applies given function to each image"""
    new_folder_path = '{}{}{}'.format(folder_path, '-', str(new_folder_name))
    while True:
        try:
            shutil.copytree(folder_path, new_folder_path)
        except FileExistsError: # If the new folder name exists, throw error
            print('The following folder already exists: {}.\n It has already been processed'.format(new_folder_name))
            break
        else:
            print('Copying: {}'.format(folder_path))
            print('New folder is : {}'.format(new_folder_path))
            break

def image_adaptive_equalization(image_directories):
    """ Computes hostograms of distinct areas of image and used them to redistribute lightness"""
    images_read = io.imread(image_directories)  # Reading in files from path
    image_altered = exposure.equalize_adapthist(images_read)
    io.imshow(image_altered)  #Show image
    io.imsave(image_directories, image_altered)  # NOTE:  Overides original image

def image_equalization(image_directories):
    """ Spreads out most frequent intensity values"""
    images_read = io.imread(image_directories)
    image_altered = exposure.equalize_hist(images_read)
    io.imshow(image_altered)
    io.imsave(image_directories, image_altered)

def image_contrast_stretching(image_directories):
    """Maps intesntity raneg between given percentiles"""
    images_read = io.imread(image_directories)
    p2, p98 = np.percentile(images_read, (2, 98))
    image_altered = exposure.rescale_intensity(images_read, in_range=(p2, p98))
    io.imshow(image_altered)
    io.imsave(image_directories, image_altered)


# Note:  For function, expect 'equalize', 'adaptive_equalization' or 'contrast_stretching'
# new_name is string
def image_processor(folder_path, new_name, function):
    """Copies folder, and applies given function to each image"""
    copy_folder_tree(folder_path, new_name)
    new_folder = '{}{}{}'.format(folder_path, '-', new_name)
    image_paths = get_image_paths(new_folder)
    if function == 'equalize':
        for image in image_paths:
            image_equalization(image)
    elif function == 'adaptive_equalization':
        for image in image_paths:
            image_adaptive_equalization(image)
    elif function == 'contrast_stretching':
        for image in image_paths:
            image_contrast_stretching(image)
    else:
        print('not valid function')



# FOLDER_TEST = '../labels/test'
# FOLDER_TRAIN = '../labels/train'

# image_processor(FOLDER_TEST, 'equalized', 'equalize')
# image_processor(FOLDER_TEST, 'adaptive-equalized', 'adaptive_equalization')
# image_processor(FOLDER_TEST, 'contrast-stretching', 'contrast_stretching')

# image_processor(FOLDER_TRAIN, 'equalized', 'equalize')
# image_processor(FOLDER_TRAIN, 'adaptive-equalized', 'adaptive_equalization')
# image_processor(FOLDER_TRAIN, 'contrast-stretching', 'contrast_stretching')
