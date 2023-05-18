import requests
import os

# Extracts images from coco url
def download_images(images_df, images_path):
    print(f'Extracting {len(images_df)} images\n')

    for count, (index, row) in enumerate(images_df.iterrows()):

        # Fetch data
        img_data = requests.get(row.coco_url).content

        # Write to path
        with open(os.path.join(images_path, row.file_name), 'wb') as handler:
            handler.write(img_data)

        print(f'Images downloaded: {count+1}/{len(images_df)}')
