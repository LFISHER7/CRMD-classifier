import os
import glob
import random
import numpy as np
from shutil import copyfile
from tqdm import tqdm

k = 10

for fold in tqdm(range(k)):
    file_list = glob.glob("../labels/train-equalized/**/*.JPG", recursive=True)
    random.seed(42)
    random.shuffle(file_list)

    num_validation_samples = len(file_list)//10

    validation_files = file_list[num_validation_samples*fold: num_validation_samples * (fold+1)]
    training_files = np.concatenate((file_list[:num_validation_samples*fold], file_list[num_validation_samples * (fold + 1):]))

    fold_dir = f"../labels_by_fold/{fold}"
    if not os.path.exists(fold_dir):
        os.mkdir(fold_dir)
        os.mkdir(f"{fold_dir}/train")
        os.mkdir(f"{fold_dir}/test")

    for validation_file in validation_files:
        label = os.path.basename(os.path.dirname(validation_file))
        label_dir = f"{fold_dir}/test/{label}"
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        copyfile(validation_file, f"{label_dir}/{os.path.basename(validation_file)}")

    for training_file in training_files:
        label = os.path.basename(os.path.dirname(training_file))
        label_dir = f"{fold_dir}/train/{label}"
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        copyfile(training_file, f"{label_dir}/{os.path.basename(training_file)}")
