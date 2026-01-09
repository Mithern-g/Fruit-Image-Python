import Seq_EdgeDetection as edt
import os
import cv2

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
print("Loaded")
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
print("Pretrained")
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights
print("Transfered")

# Train the model on your custom dataset
model.train(data="data.yaml", epochs=100, imgsz=480,  device=0)
print("Ready")
#edt.Edge_Get('FruitExtracted/Apple-Fresh/IMG-20240612-WA0025.jpg')

image_target = "file.jpg"

def Generate_Dataset():
    directory = os.fsencode("FruitExtracted")

    for folder in os.listdir(directory):
        folder_name = folder.decode('utf-8')
        for files in os.listdir(os.path.join("FruitExtracted",folder_name)):
            #Comically jank method of getting file path in relation to python execution folder
            files_name = str(files)
            newfile = str(os.path.join(os.path.join("FruitExtracted",folder_name),files_name))
            newfile_edit = edt.Edge_Get(newfile)

            newpath = os.path.join("FruitNew",folder_name)
            if not os.path.isdir(newpath):
                os.makedirs(newpath)
            new_file_name = str(os.path.join(newpath,files_name))
            cv2.imwrite(new_file_name,newfile_edit)

    return
    #None

def GetImage():
    processed = edt.Edge_Get(image_target)

    result = model(source=image_target, show=True, conf=0.4, save=True)

    return

#Generate_Dataset()
GetImage()
