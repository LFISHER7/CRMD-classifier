import numpy as np
from PIL import Image
from mean_image import get_image_paths


def load_image(infilename):
    """Loads image from path"""
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def data_as_numpy(train_or_test, size, name):
    """Enter string of train or test
       Creates numpy array of correct size for given model
       NOTE: Need to create folder manually
       """
    if size == 227:
        ar = ''
    elif size == 150:
        ar = '_150'
    else:
        print('Not correct size')
        raise ValueError


    paths = get_image_paths('../labels/{}-equalized{}'.format(train_or_test, ar))
    arrays_list = []
    for image in paths:
        array = load_image(image)
        arrays_list.append(array)

    result = np.array(arrays_list)
    print('Data shape: {}'.format(result.shape))


    np.save('../Data_as_numpy/{}_MODEL/{}.npy'.format(name, train_or_test), result)

def labels_as_numpy(train_or_test, size, name):
    """Creates labels - 1D array"""
    if size == 227:
        ar = ''
    elif size == 150:
        ar = '_150'
    else:
        print('Not correct size')
        raise ValueError

    paths = (get_image_paths('../labels/{}-equalized{}'.format(train_or_test, ar)))

    models = []
    for path in paths:
        model = path.split('/')[-2]
        models.append(model)

    labels = {}
    model_labels = sorted(set(models))

    i = 0      #Was starting at 1 but then index out of range when fitting model, change to 0.
    for model in model_labels:
        labels[model] = i
        i += 1

    #now want to iterate through list and if equals key give i t that value.
    number_labels = []
    for model in models:
        if model in labels.keys():
            model = labels[model]
            number_labels.append(model)

    label_array = np.array(number_labels)
    print('Label shape: {}'.format(label_array.shape))


    np.save('../Data_as_numpy/{}_MODEL/{}-label.npy'.format(name, train_or_test), label_array)



data_as_numpy('train', 150, 'VGG16')
data_as_numpy('test', 150, 'VGG16')
data_as_numpy('train', 227, 'DENSE')
data_as_numpy('test', 227, 'DENSE')

labels_as_numpy('train', 150, 'VGG16')
labels_as_numpy('test', 150, 'VGG16')
labels_as_numpy('train', 227, 'DENSE')
labels_as_numpy('test', 227, 'DENSE')
