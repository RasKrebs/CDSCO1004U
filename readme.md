# `CDSCO1004E FINAL EXAM REPO`

This is the repository for all python code, models and related files needed for the final exam CDSCO1004E.

The directory is structured as follows:
```
main/
├── setup.ipynb
├── CNN_training.ipynb
├── CNN_evaluation.ipynb
├── YOLOv5_training.ipynb
├── YOLOv5_evaluation_detection.ipynb
├── dependencies/
│   ├── cnn_data/
│   │   ├── test
│   │   └── training
│   ├── models/
│   │   ├── TinyVGG_Model.keras
│   │   └── Xception_Model.keras
│   ├── yolo_data/
│   │   └── coco2017/
│   │       ├── instances_train2017.json
│   │       └── instances_val2017.json
│   └── yolov5/
│       └── yolo_model_directory...
├── utils/
│   ├── DataLoader.py
│   ├── FetchImages.py
│   ├── GenerateFileList.py
│   ├── ImageJsonGenerator.py
│   ├── ImageModifier.py
│   ├── MoveData.py
│   └── YOLOLabelGenerator.py
└── model_architecture/
    ├── tiny_vgg_architecture.png
    └── Xception_architecture.png
```

Within the main folder, one will find 5 notebooks:
1. `setup.ipynb`: This contains everything necessary for extracting and transforming the data, needed for modelling.
2. `CNN_training.ipynb`: It is within this notebook the training of TinyVGG and the simplified Xception is happening. *Note, these models are already trained and saved within the dependencies/models folder*
3. `CNN_evaluation.ipynb`: This is for evaluating the models trained within the `CNN_training.ipynb` notebook.
4. `YOLOv5_training.ipynb`: Notebook for training the YOLOv5 model.
5. `YOLOv5_evaluation_detection.ipynb`: Notebook for evaluating and performing detection using the custom trained YOLOv5 model.

Within dependencies, data for the CNN models are stored, instance JSON files for the YOLOv5 model, and the YOLOv5 directroy. *Note, that images for for YOLOv5 is found within the YOLOv5 directory under datasets*

The remaining elements in the folder contain general functions used throughout the project.
