# Library import
import json
import os
import pandas as pd
from sklearn.utils import Bunch
import requests
 
from utils.GenerateFileList import unpack_json, balanced_category_sampling
from utils.FetchImages import download_images
from utils.ImageJsonGenerator import create_subset_json
from utils.YOLOLabelGenerator import generate_txt_files

# Necessary Paths
WORKING_DIRECTORY = os.getcwd()
DEPENDENCIES = os.path.join(WORKING_DIRECTORY, 'dependencies')
DATA_PATH = os.path.join(DEPENDENCIES, 'yolo_data')
COCO_ANNOTATIONS = os.path.join(DATA_PATH, 'coco2017')

TRAINING_PATH = os.path.join(DATA_PATH, 'training')
TRAINING_IMAGES = os.path.join(TRAINING_PATH, 'images')
TRAINING_DATA = os.path.join(TRAINING_PATH, 'data')
TRAINING_LABEL = os.path.join(TRAINING_PATH, 'labels')

TEST_PATH = os.path.join(DATA_PATH, 'test')
TEST_IMAGES = os.path.join(TEST_PATH, 'images')
TEST_DATA = os.path.join(TEST_PATH, 'data')
TEST_LABEL = os.path.join(TEST_PATH, 'labels')

VALIDATION_PATH = os.path.join(DATA_PATH, 'validation')
VALIDATION_IMAGES = os.path.join(VALIDATION_PATH, 'images')
VALIDATION_DATA = os.path.join(VALIDATION_PATH, 'data')
VALIDATION_LABELS = os.path.join(VALIDATION_PATH, 'labels')

# For simplicity AND naming will collect them in a bunch Object
PATHS = Bunch(
    WORKING_DIRECTORY = WORKING_DIRECTORY,
    DEPENDENCIES = DEPENDENCIES,
    DATA_PATH = DATA_PATH,
    COCO_ANNOTATIONS = COCO_ANNOTATIONS,
    TRAINING_PATH = TRAINING_PATH,
    TRAINING_IMAGES = TRAINING_IMAGES,
    TRAINING_DATA = TRAINING_DATA,
    TRAINING_LABEL = TRAINING_LABEL,
    TEST_PATH = TEST_PATH,
    TEST_IMAGES = TEST_IMAGES,
    TEST_DATA = TEST_DATA,
    TEST_LABEL = TEST_LABEL,
    VALIDATION_PATH = VALIDATION_PATH,
    VALIDATION_IMAGES = VALIDATION_IMAGES,
    VALIDATION_DATA = VALIDATION_DATA,
    VALIDATION_LABELS = VALIDATION_LABELS,
)

CATEGORIES = ["traffic light", "bus", "train", "truck", "car", "bicycle", "person"]

# Create all necessary directories if not exists
print('(1/8): Creating Directory Structure...')
for p in PATHS.values():
    if not os.path.exists(p):
        print(f'{os.path.basename(p)} does not exists')
        os.makedirs(p)
print('Done')

# Extract instances json from web if not exists
print('(2/8): Downloading COCO annotations...')
# Download coco annotations from
if not os.path.exists(os.path.join(PATHS.COCO_ANNOTATIONS, 'instances_train2017.json')):
    print('Downloading training annotations...')
    URL = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_train2017.json"
    train_annotation_url = requests.get(URL).content

    with open(os.path.join(PATHS.COCO_ANNOTATIONS, 'instances_train2017.json'), "wb") as file:
        file.write(train_annotation_url)
    print('Done')
else:
    print('Training annotations already exists')

if not os.path.exists(os.path.join(PATHS.COCO_ANNOTATIONS, 'instances_val2017.json')):
    print('Downloading validation annotations...')
    URL = "https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_val2017.json"
    train_annotation_url = requests.get(URL).content

    with open(os.path.join(PATHS.COCO_ANNOTATIONS, 'instances_val2017.json'), "wb") as file:
        file.write(train_annotation_url)
    print('Done')
else:
    print('Validation annotations already exists')


# Unpack instance json and generate list of useful files
print('(3/8): Fetching Files...')
train_files, train_data = unpack_json(labels=CATEGORIES, 
                                      annotation_file_name='instances_train2017.json',
                                      max_img_categories=3, 
                                      annotation_path=PATHS.COCO_ANNOTATIONS)
print('Training files fetched')

val_files, val_data = unpack_json(labels=CATEGORIES, 
                                  annotation_file_name='instances_val2017.json',
                                  max_img_categories=3, 
                                  annotation_path=PATHS.COCO_ANNOTATIONS)
print('Validation files fetched')
print('(3/8): Done')


# Balance samples evenly across categories
print('(4/8) Balancing distributed sampling across categories')
train_images, _ = balanced_category_sampling(files = train_files,
                                          data = train_data,
                                          size = 2500,
                                          categories = CATEGORIES)

val_images, _ = balanced_category_sampling(files = val_files,
                                          data = val_data,
                                          size = 500,
                                          categories = CATEGORIES)



print('(4/8): Done')

# Download images
print('(5/8): Downloading images...')
download_images(train_images, PATHS.TRAINING_IMAGES)
download_images(val_images, PATHS.VALIDATION_IMAGES)

print('(5/8): Done')


print('(6/8): Generating COCO JSON for training and test images...')
create_subset_json(data = train_data,
                   file_name='train',
                   image_path = PATHS.TRAINING_IMAGES,
                   data_path= PATHS.TRAINING_DATA) # Training JSON
print('Train done')

create_subset_json(data = val_data,
                   file_name='validation',
                   image_path = PATHS.VALIDATION_IMAGES,
                   data_path = PATHS.VALIDATION_DATA) # TEST JSON
print('Valiation done')
print('(6/8): Done')

print('(7/8) Resizing images...')
resize_images(PATHS.TRAINING_IMAGES)
resize_images(PATHS.VALIDATION_IMAGES)


print('(8/8): Creating YOLO labels...')
generate_txt_files(data_path=PATHS.TRAINING_DATA,
                   img_path=PATHS.TRAINING_IMAGES,
                   label_path=PATHS.TRAINING_LABEL,
                   categories=CATEGORIES,
                   data_filename='train.json')
print('Training done')

generate_txt_files(data_path=PATHS.VALIDATION_DATA,
                   img_path=PATHS.VALIDATION_IMAGES,
                   label_path=PATHS.VALIDATION_LABELS,
                   categories=CATEGORIES,
                   data_filename='validation.json')
print('Validation done')
print('(8/8): Done')
