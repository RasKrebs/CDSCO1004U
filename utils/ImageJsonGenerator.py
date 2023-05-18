from sklearn.utils import Bunch
import os
import json
import pandas as pd 

def create_subset_json(data,
                       image_path,
                       data_path,
                       file_name,
                       file_data = None) -> Bunch:
    
    # File_data parameter to ensure json can be updated if need be
    files_in_dir = os.listdir(image_path)
    
    if file_data == None:
        file_ids = [x['id'] for x in data['images'] if x['file_name'] in files_in_dir]
    else:
        file_ids = [file_data[filename] for filename in files_in_dir]
    
    
    # Generating list of images and annotations for images
    images = [x for x in data['images'] if x['id'] in file_ids]
    
    # Extracting annotations using pandas
    annot_df = pd.DataFrame(data['annotations'])
    index_of_img = list(annot_df[annot_df.image_id.isin(file_ids)].index)
    annotations = [data['annotations'][i] for i in index_of_img]
    
    del annot_df
    del index_of_img
    
    # Creating Bunch object
    out = Bunch(
        info=data['info'],
        licenses=data['licenses'],
        images=images,
        annotations=annotations,
        categories=data['categories'])

    # Serializing json
    json_object = json.dumps(out)
    
    # Writing json        
    with open(f'{data_path}/{file_name}.json', "w") as outfile:
        outfile.write(json_object)
        