import os

ROOT_DIR = os.path.dirname(os.path.abspath('../..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data')
COCO_BASE_PATH = os.path.join(DATA_PATH, 'coco2017')
COCO_TRAIN_PATH = os.path.join(COCO_BASE_PATH, 'train2017')
NEW_TRAINING_PATH = os.path.join(DATA_PATH, 'training')
NEW_TEST_PATH  = os.path.join(DATA_PATH, 'test')
NEW_TRAINING_IMAGES_PATH = os.path.join(NEW_TRAINING_PATH, "images")
NEW_TEST_IMAGES_PATH = os.path.join(NEW_TEST_PATH, "images")
NEW_TRAINING_DATA_PATH = os.path.join(NEW_TRAINING_PATH, "data")
NEW_TEST_DATA_PATH = os.path.join(NEW_TEST_PATH, "data")


def FactoryReset(new_image_path,
                 new_data_path,
                 coco_image_path):
    try:
        image_files = os.listdir(new_image_path)
    except:
        image_files = []
        
    try:
        data_file = os.listdir(new_data_path)[0]
        os.remove(os.path.join(new_data_path, data_file))
    except:
        pass
    
    for file in image_files:
        os.rename(os.path.join(new_image_path, file),
                  os.path.join(coco_image_path, file)
                  )
        
FactoryReset(NEW_TRAINING_IMAGES_PATH, NEW_TRAINING_DATA_PATH, COCO_TRAIN_PATH)
FactoryReset(NEW_TEST_IMAGES_PATH, NEW_TEST_DATA_PATH, COCO_TRAIN_PATH)