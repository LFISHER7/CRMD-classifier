import os
import csv
import plotly as py
import plotly.graph_objs as go
py.tools.set_credentials_file(username='LFISHER77', api_key='CJuRli36Pt6EP5V4f8P5')


def get_file_paths(folder):
    """Gets path of every file in folder"""
    names = []
    file_paths = []
    for file in os.listdir(folder):
        if not file.startswith('.'): #Getting rid of .DS_Store
            names.append(file)
            file_path = os.path.join(folder, file)
            file_paths.append(file_path)
    return file_paths, names

def open_csv(path, x_data, y_data):
    """Opens csv and extracts relevant columns"""
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        epochs = []
        accuracy = []
        for column in readCSV: #Dont want first column - contains wall time
            epochs.append(column[1])
            accuracy.append(column[2])
        del epochs[0]
        del accuracy[0]
        y_data.append(accuracy)
        x_data.append(epochs)

def get_data(folder):
    """Opens csv file and extracts data - appends to lists"""
    x_data = []
    y_data = []
    paths, names = get_file_paths(folder)
    paths.sort() # Sort so plot in order
    for path in paths:
        open_csv(path, x_data, y_data)
    return names, y_data, x_data


def trace(index, name, line_type, colour_index, x_data, y_data):
    """Generates trace by accessing data at index"""
    colours = ['rgb(205, 12, 24)', 'rgb(30, 144, 255)', 'rgb(0,0,205)',
              'rgb(133, 189, 0)', 'rgb(238, 189, 103)', 'rgb(219, 113, 31)', 'rgb(255, 16, 123)',
              'rgb(156, 16, 123)', 'rgb(11, 241, 123)'] #RGB colour spaces
    trace = go.Scatter(
        x=x_data[index],
        y=y_data[index],
        mode='lines',
        name=name.replace('.csv', ''), #Otherwise have csv on end
        line=dict(color=(colours[colour_index]),
        width=2,
        dash=line_type))
    return trace

# NAMES, Y_DATA, X_DATA = get_data('../tensorboard_csv/transfer_summary/accuracy')
# NAMES.sort()

# TRACE1 = trace(0, NAMES[0], 'none', 0, X_DATA, Y_DATA)
# TRACE2 = trace(1, NAMES[1], 'none', 1, X_DATA, Y_DATA)
# TRACE3 = trace(2, NAMES[2], 'none', 2, X_DATA, Y_DATA)

# TRACE4 = trace(3, NAMES[3], 'dot', 0, X_DATA, Y_DATA)
# TRACE5 = trace(4, NAMES[4], 'dot', 1, X_DATA, Y_DATA)
# TRACE6 = trace(5, NAMES[5], 'dot', 2, X_DATA, Y_DATA)

# DATA = [TRACE1, TRACE2, TRACE3, TRACE4, TRACE5, TRACE6]

# LAYOUT = dict(title='Transfer Learning Summary - Accuracy',
#               xaxis=dict(title='Number of Epochs', showgrid=True),
#               yaxis=dict(title='Accuracy', showgrid=True),
#               )
# FIG = dict(data=DATA, layout=LAYOUT)
# py.plotly.iplot(FIG, filename='Transfer Summary - ACC')

# print('Graph plotted on plotly')
