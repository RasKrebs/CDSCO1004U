import os
import pandas as pd
from .DataLoader import import_data


def generate_annot_df(data_path, categories, data_filename):
    
    bunch = import_data(data_path, file_name=data_filename)
    # Generating Annotation df with YOLO format bbox
    
    dic = {x['id']: x['file_name'] for x in bunch.images}
    category_ids = {cat['id']: cat['name'] for cat in bunch.catagories if cat['name'] in categories}
    
    df = pd.DataFrame(bunch.annotations)[['image_id', 'bbox', 'category_id']]
    
    df['file_name'] = df['image_id'].map(dic)
    
    df = df.assign(
        bbox0 = lambda x: x['bbox'].apply(lambda x: x[0]),
        bbox1 = lambda x: x['bbox'].apply(lambda x: x[1]),
        bbox2 = lambda x: x['bbox'].apply(lambda x: x[2]),
        bbox3 = lambda x: x['bbox'].apply(lambda x: x[3]),
        center_x = lambda x: x['bbox0'] + x['bbox2'] / 2,
        center_y = lambda x: x['bbox1'] + x['bbox3'] / 2,
        width = lambda x: 2*(x['center_x'] - x['bbox0']),
        height = lambda x: 2*(x['center_y'] - x['bbox1'])
    )
    
    df = df.reset_index(drop=True)
    df = df[df.category_id.isin(category_ids.keys())][['file_name', 'image_id','category_id', 'center_x','center_y', 'width', 'height']]
    
    # Creating new category ids
    categories = {old: new for (new, old) in zip(range(len(category_ids.keys())), category_ids.keys())}
    df.category_id = df.category_id.map(categories)
    
    return df


def generate_txt_files(data_path, data_filename, img_path, label_path, categories):
    
    df = generate_annot_df(data_path=data_path,
                           categories=categories, data_filename=data_filename)
    
    files = os.listdir(img_path)
    print(f'Generating {len(files)} text files')
    for file in files:
        print(f'File number: {files.index(file)+1}/{len(files)}')
        if file == '.DS_Store':
            continue

        ph = df[df.file_name == file]

        if len(ph) > 1:
            lines = []
            for ind, row in ph.iterrows():
                x_center_n = (row.center_x)/640
                y_center_n = (row.center_y)/640
                width_n = (row.width)/640
                height_n = (row.height)/640

                line = ' '.join((str(row.category_id), str(x_center_n), str(
                    y_center_n), str(width_n), str(height_n))) + '\n'
                lines.append(line)

            with open(os.path.join(label_path, f"{file.strip('.jpg')}.txt"), 'w+') as file:
                for line in lines:
                    file.write(line)
        else:
            print(file)
            row_dic = ph.to_dict('records')[0]
            x_center_n = (row_dic['center_x'])/640
            y_center_n = (row_dic['center_y'])/640
            width_n = (row_dic['width'])/640
            height_n = (row_dic['height'])/640
            line = ' '.join((str(row_dic['category_id']), str(x_center_n), str(
                y_center_n), str(width_n), str(height_n))) + '\n'
            with open(os.path.join(label_path, f"{file.strip('.jpg')}.txt"), 'w+') as file:
                file.write(line)
