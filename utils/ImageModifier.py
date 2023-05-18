import os
from PIL import Image, ImageOps

def resize_images(path):
    # Find images in image path
    images = [os.path.join(path, file) for file in os.listdir(path) if '.jpg' in file]
    
    print(f'Resizing {len(images)} number of images:\n')
    
    # loop through images
    for image in images:
        # Print iteration
        print(f'Image: {images.index(image)+1}/{len(images)}')
        
        # Open image
        with Image.open(image) as img:
            width, height = img.size

            width_padding = max(0, (640 - width))
            height_padding = max(0, (640 - height))
            
            # Add padding, so image i 640 by 640. is done by centering to 0,0, meaning bounding boxes are still applicable
            padding = (0, 0, width_padding, height_padding)
            img = ImageOps.expand(img, padding)

            
            # Save image
            img.save(image)