from PIL import Image
from mean_image import get_image_paths

def get_image_size(image_path):
    """Opens image given image path and returns its dimensions"""
    im = Image.open(image_path)
    image_size = im.size
    return image_size

def resize_to_227(folder_path):
    """Gets all image paths downstream of folder and checks size.  If not (227, 227), resizes"""
    image_paths = get_image_paths(folder_path) # from mean_image.py
    for image in image_paths:
        size = get_image_size(image)
        print(size)
        if size == (227, 227):
            print(image)
            print('Image is the right size')
        else:
            img = Image.open(image)
            im_resized = img.resize((227, 227)) #Most of my images are 227, 227
            im_resized.save(image)



# TRAIN_PATH = '../labels/train/'
# resize_to_227(TRAIN_PATH)
