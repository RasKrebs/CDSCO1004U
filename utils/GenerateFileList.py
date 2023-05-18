import json
import pandas as pd
    
def unpack_json(labels, annotation_path, annotation_file_name, crowd = False, max_img_categories = 3):
    """Function for collecting data from coco2017 train folder

    Params:
    * labels: List of categories to filter
    * crowd: Include images of crowds or not (default False)
    * paths: Bunch object with relevant paths
    * max_img_categories: Specify max number of categories in image (default None)
    
    Returns:
        _type_: dict
    """
    
    # Fetching JSON
    coco_instances = open(f'{annotation_path}/{annotation_file_name}')
    data = json.load(coco_instances)

    # Extracting category IDs
    category_ids = [c['id'] for c in data['categories'] if c['name'] in labels]

    # Filtering irrelevant data
    
    # Extracting unique image ids that is within one of the categories
    annotations = pd.DataFrame(data['annotations'])
    image_ids = annotations[annotations.category_id.isin(category_ids)].image_id.unique()

    
    # Removing images with crowds if specified by param input + images with too many categories
    if not crowd:
        # Fetching image id of images with crowds
        image_with_crowds = annotations[annotations.iscrowd == 1].image_id.unique()
        
        if max_img_categories == None:
            annotations = (annotations[(annotations.image_id.isin(image_ids)) & 
                                    (-annotations.image_id.isin(image_with_crowds))])
        
        else:
            annotations = (annotations[(annotations.image_id.isin(image_ids)) & 
                                    (-annotations.image_id.isin(image_with_crowds))]
                    .groupby('image_id').filter(lambda x: len(x) <= max_img_categories))
    else:
        if max_img_categories == None:
            annotations = (annotations[(annotations.image_id.isin(image_ids))])

        else:
            annotations = (annotations[(annotations.image_id.isin(image_ids))].
                           groupby('image_id').filter(lambda x: len(x) <= max_img_categories))
        
        
        annotations = (annotations[(annotations.image_id.isin(image_ids)) & 
                                (-annotations.image_id.isin(image_with_crowds))]
                    .groupby('image_id').filter(lambda x: len(x) <= max_img_categories))

    # Garbage-collecting image_with_crowds if used
    try:
        del image_with_crowds
    except:
        pass
    
    # Updaing image id variable
    image_ids = annotations.image_id.unique()

    
    # Fetch filenames
    file_placeholder = {image['id']: image['file_name']
                        for image in data['images']}
    
    files_with_ids = {file_placeholder[id]: id for id in image_ids}
    del file_placeholder
    
    return files_with_ids, data


def balanced_category_sampling(files,
                               data,
                               size,
                               categories,
                               list_of_files_to_exclude = None):

    # Id of categories
    requested_cats = [cat['id'] for cat in data['categories'] if cat['name'] in categories]

    # Annotation datafraem for image annotations
    annot_df = pd.DataFrame(data['annotations'])
    
    if list_of_files_to_exclude != None:
        files = {k: v for k, v in files.items() if k not in list_of_files_to_exclude}
    
    annot_df = annot_df[annot_df.image_id.isin(files.values())]

    # Function for calculating the distributed value per category (distributes size input as evenly as possible acros categories)
    def distribute(size, units):
        base, extra = divmod(size, units)
        return [base + (i < extra) for i in range(units)]

    # Dictionary with distribution assigned to id
    images_per_class = {id: nr for (id, nr) in zip(
        requested_cats, distribute(size=size, units=len(categories)))}

    # Filter annotations by first removing all annotations which are not within desired categories. Following this we sourt by area, to find the most prominent bbox in image, and dropping any subsequent annotations. From these we then filter so that we pick images according to the distribution defined earlier
    annot_df = annot_df[annot_df.category_id.isin(requested_cats)].sort_values(by='area', ascending=False).drop_duplicates(subset='image_id').groupby('category_id').apply(lambda x: x[:images_per_class[x.category_id.values[0]]]).reset_index(drop=True)
    image_id_distributed = annot_df.image_id.to_list()
    # The purpose with this is from the images distribute classes evenly. The output is unique images, where the most prominent bbox should be within each category. Should ensure better representation across images
    images = pd.DataFrame(data['images'])
    return images[images.id.isin(image_id_distributed)], annot_df
