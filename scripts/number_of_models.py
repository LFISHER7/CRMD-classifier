import os
import plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
py.tools.set_credentials_file(username='LFISHER77', api_key='CJuRli36Pt6EP5V4f8P5')


def get_number_of_files(path_to_folder):
    """Goes into folder and add number of files to dinctionary with filename as key"""
    file_numbers = {}
    file = [f for f in os.listdir(path_to_folder) if not f.startswith('.')]   #Removes .DS_Store
    file_len = len(file)
    folder_name_split = path_to_folder.split('/')# splits it at every /, stores as list
    folder_name = folder_name_split[-1]       #Folder name is last element of above list.
    file_numbers[folder_name] = file_len #Adding to dictionary
    return file_numbers

def get_file_paths(folder):
    """Gets path of every file in folder"""
    file_paths = []
    for file in os.listdir(folder):
        if not file.startswith('.'):
            file_path = os.path.join(folder, file) #Joins with '/'
            file_paths.append(file_path)
    return file_paths


def number_of_models(directory):
    """Gets path of every folder and counts number of files in each"""
    file_paths = get_file_paths(directory)  #Calling function to get paths
    values = {}    #Create new dictionary which can be updated with generated key:value pairs
    for file in file_paths:
        dicts = get_number_of_files(file)
        values.update(dicts)
    return values


# print('This is the number of each model in the training data set:')
# TRAIN_DATA = number_of_models('../labels/train')
# print(TRAIN_DATA)
# print()
# print('This is the number of each model in the test data set:')
# TEST_DATA = number_of_models('../labels/test')
# print(TEST_DATA)


#####PLOTTING WITH PLOTLY)#####

# TRACE_1 = go.Bar(
#     x=list(TRAIN_DATA.keys()),
#     y=list(TRAIN_DATA.values()),
#     name='Train'
# )

# TRACE_2 = go.Bar(
#     x=list(TEST_DATA.keys()),
#     y=list(TEST_DATA.values()),
#     name='test'
# )

# DATA = [TRACE_1, TRACE_2]
# LAYOUT = go.Layout(
#     barmode='grouped'
# )

# FIG = go.Figure(data=DATA, layout=LAYOUT)
# py.plotly.iplot(FIG, filename='stacked-bar')


# ######PLOTTING WITH MATPLOTLIB######

# PLOT_TRAIN = plt.bar(TRAIN_DATA.keys(), TRAIN_DATA.values(), width=0.5, color='b')
# PLOT_TRAIN = plt.bar(TEST_DATA.keys(), TEST_DATA.values(), width=0.5, color='r')

# plt.title('Distribution of different models in training and test sets')
# plt.ylabel('Number of each model')
# plt.legend((PLOT_TRAIN[0], PLOT_TEST[0]), ('Train', 'Test'))
# plt.xticks(rotation=90) #Change rotation of x labels
# plt.show()
