from sklearn.utils import Bunch
import os
import json

def import_data(data_path, file_name) -> Bunch:
    """Reads data jsons from paths. Outputs results as sklearn Bunch"""
    
    with open(os.path.join(data_path, file_name), 'r') as openfile:
        # Reading from json file
        o = json.load(openfile)
        
    return Bunch(
        info=o['info'],
        licenses=o['licenses'],
        images=o['images'],
        annotations=o['annotations'],
        catagories=o['categories']
        )