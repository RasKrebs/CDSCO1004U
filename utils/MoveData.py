import os
from sklearn.utils import Bunch

def generate_data_subsets(paths, 
                          list_of_files: list, 
                          training_size=2000, 
                          split=0.75):
    """
    This function simply takes care of moving files from the COCO 2017 into our own directory, to reduce the need for accessing the large amounts of files in that folder.

    Params:
    * paths [Bunch]: Bunch object with containing all relevant paths
    * list_of_files [list]: List of files that should be moved. These should be filtered prior so that to only include images within relevant category
    * training_size [int]: Number of images to include in training
    * split [float]: Test train split:

    """

    test_size = round(training_size * (1 - split))
    
    if len(list_of_files) < (test_size+training_size):
        raise IndexError('Insufficient number of files for training and test')

    
    # Create directory if not exists
    for dir in [paths.NEW_TRAINING_PATH, paths.NEW_TEST_PATH, paths.NEW_TRAINING_IMAGES_PATH, paths.NEW_TEST_IMAGES_PATH, paths.NEW_TRAINING_DATA_PATH, paths.NEW_TEST_DATA_PATH, paths.NEW_VALIDATION_PATH, paths.NEW_VALIDATION_IMAGES_PATH, paths.NEW_VALIDATION_DATA_PATH]:
        if not os.path.exists(dir):
            os.makedirs(dir)
            
            
    # Extracting existing files
    train_existing_files = os.listdir(paths.NEW_TRAINING_IMAGES_PATH)
    test_existing_files = os.listdir(paths.NEW_TEST_IMAGES_PATH)
    val_existing_files = os.listdir(paths.NEW_VALIDATION_IMAGES_PATH)
    
    if (len(train_existing_files) >= training_size) & (len(test_existing_files)+len(val_existing_files) >= test_size):
        print(f"Number of images in training and test is already equal to the inputted amount")
        return

    elif (len(train_existing_files) < training_size) & (len(test_existing_files)+len(val_existing_files) >= test_size):
        print(f"Number of images in test is sufficient. Ingesting {training_size-len(train_existing_files)} images to training")
        training_size = training_size - len(train_existing_files)
        test_size = 0

    elif (len(train_existing_files) >= training_size) & (len(test_existing_files)+len(val_existing_files) < test_size):
        print(f"Number of images in training is sufficient. Ingesting {test_size-len(test_existing_files)} images to test")
        test_size = test_size - len(test_existing_files)+len(val_existing_files)
        training_size = 0
    else:
        print(f"Ingesting {test_size-len(test_existing_files)+len(val_existing_files)} images to test\nIngesting {training_size-len(train_existing_files)} images to training")
        test_size = test_size - len(test_existing_files)+len(val_existing_files)
        training_size = training_size - len(train_existing_files)
    
    # Extracting image file names
    print("Extracting filenames\n")
    files = [file for file in list_of_files if file not in (train_existing_files + test_existing_files)]
    
    # Generating lists for training and testing
    training_files = files[:training_size]
    files_with_out_training_files = [file for file in files if file not in training_files]
    test_files = files_with_out_training_files[:int(test_size)]
    del files_with_out_training_files
    
    # Moving files
    print(f"Moving {training_size} training images\n")
    for file in training_files:
        print(f"File number: {training_files.index(file)}/{training_size}")
        os.rename(
            os.path.join(paths.COCO_TRAIN_PATH, file),
            os.path.join(paths.NEW_TRAINING_IMAGES_PATH, file),
        )
    print(" ")

    print(f"Moving {test_size} test images\n")
    for file in test_files:
        print(f"File number: {test_files.index(file)+1}/{test_size}")
        os.rename(
            os.path.join(paths.COCO_TRAIN_PATH, file),
            os.path.join(paths.NEW_TEST_IMAGES_PATH, file),
        )
