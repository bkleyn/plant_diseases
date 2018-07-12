import os
import numpy as np
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

INPUT_SIZE = 256

SET = [0.7, 0.5, 0.3]
for TRAIN_SIZE in SET:

    # Set input directory
    data_dir = "../data/segmented/"

    # Get classes
    classes = os.listdir(data_dir + 'original/')
    num_classes = len(classes)
    print(f"Number of classes: {num_classes}")

    # Create train/validation directories
    name = str(int(TRAIN_SIZE * 100)) + '_' + str(int((1-TRAIN_SIZE) * 100))
    os.mkdir(data_dir + name)

    train_dir = data_dir + name + '/train/'
    val_dir = data_dir + name + '/validation/'

    os.mkdir(train_dir)
    os.mkdir(val_dir)

    # Create new class directories for train and validation images
    for c in classes:
        os.mkdir(train_dir + c)
        os.mkdir(val_dir + c)

    # Get path and label for each image
    db = []
    for label, class_name in enumerate(classes):
        path = data_dir + 'original/' + class_name
        for file in os.listdir(path):
            if '.ini' not in file:
                db.append(['{}/{}'.format(class_name, file), label, class_name])
    db = pd.DataFrame(db, columns=['file', 'label', 'class_name'])
    num_images = len(db)
    print(f"Number of images: {num_images}")

    # Sample train/test observations
    np.random.seed(87)
    msk = np.random.binomial(1, TRAIN_SIZE, num_images)

    # Import images
    i = 0
    for file in tqdm(db['file'].values):

        from_path = data_dir + 'original/' + file
        if msk[i] == 1:
            to_path = train_dir + file
        else:
            to_path = val_dir + file

        copyfile(from_path, to_path)
        i += 1

