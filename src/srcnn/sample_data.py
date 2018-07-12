"""Randomly sample subset of images to use for SRCNN model."""

import os
import numpy as np
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

TRAIN_SIZE = 0.01

# Set input directory
data_dir = "../../data/raw/plant_diseases/color/"

# Get classes
classes = os.listdir(data_dir + 'original/')
if '.DS_Store' in classes: # Added by BR on 4/19 to enhance compatibility with Mac
    classes.remove('.DS_Store')
num_classes = len(classes)

# Get path and label for each image
db = []
for label, class_name in enumerate(classes):
    path = data_dir + 'original/' + class_name
    for file in os.listdir(path):
        if '.ini' not in file and '.DS_Store' not in file:
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

    index = file.find("/") + 1
    to_path = "./input/" + file[index:]
    if msk[i] == 1:
        copyfile(from_path, to_path)
    i += 1
    
