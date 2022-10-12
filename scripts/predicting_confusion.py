import itertools
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, f1_score

def predict(model_path, data_path, labels_path):
    """Given model and data, makes prediction of model type"""
    model = load_model(model_path)
    data = np.load(data_path)
    labels = np.load(labels_path)

    datagen = ImageDataGenerator(rescale=1. / 255) #Dont want to augment, just rescale

    generator_predict = datagen.flow(
        data,
        batch_size=16,
        shuffle=False) #Dont shuffle otherwise labels wont match.

    predicted_labels = []
    predictions = model.predict_generator(generator_predict, verbose=1, steps=11) # 11*16 covers 163 test

    for item in predictions:
        predicted_labels.append(np.argmax(item)) #argmax returns index maximum value in softmax probabilities

    labels = list(labels)

    print('These are the predicted labels: {}'.format(predicted_labels))
    print()
    print('These are the actual labels: {}'.format(labels))

    y_true = labels
    y_pred = predicted_labels
    results = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')

    print('This is the f1 score: {}'.format(f1))
    return results, y_true, y_pred

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """Plots confusion matrix.  If want normalization, set normalize-True"""

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def predict_and_plot(model):
    """Give model name as string - computes and plits confusion matrix."""
    model_path = '../FINAL_MODELS/{}'.format(model)
    data_path = '../Data_as_numpy/{}/test.npy'.format(model)
    labels_path = '../Data_as_numpy/{}/test-label.npy'.format(model)

    indexes = {'Boston - Guidant - Autogen': 0, 'Boston - Guidant - Ingenio': 1, 'Boston - Guidant - Proponent': 2,
         'Boston - Guidant - Visionist': 3, 'Medtronic - Adapta': 4, 'Medtronic - Advisa': 5,
         'Medtronic - Claria': 6, 'Medtronic - Evera': 7, 'Medtronic - Viva': 8, 'St. Jude - Accent': 9,
         'St. Jude - Allure Quadra': 10, 'St. Jude - Ellipse': 11, 'St. Jude - Quadra Assura': 12}

    results, y_true, y_pred = predict(model_path, data_path, labels_path)

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(40, 8))
    plot_confusion_matrix(results, classes=indexes.keys(),
                      title='Confusion matrix - no normalization - {}'.format(model))

    # Plot normalized confusion matrix
    plt.figure(figsize=(40, 8))
    plot_confusion_matrix(results, classes=indexes.keys(), normalize=True,
                      title='Normalized confusion matrix - {}'.format(model))

    plt.show()

    f1 = f1_score(y_true, y_pred, average='micro')
    print('This is the f1 score: {}'.format(f1))

# predict_and_plot('VGG16_MODEL')
# predict_and_plot('DENSE_MODEL')
