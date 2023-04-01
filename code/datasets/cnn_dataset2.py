# this file contains the code to implement a custom Dataset class
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.datasets import VisionDataset
from PIL import Image
from PIL import ImageFile
# sometimes, you will have images without an ending bit # this takes care of those kind of (corrupt) images 
ImageFile.LOAD_TRUNCATED_IMAGES = True


class GalaxyImageDataset(VisionDataset):
    def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
        self.img_labels=pd.read_csv(annotations_file)
        self.root=img_dir
        self.transform=transform
        self.target_transform=target_transform
    
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, str(self.img_labels.iloc[idx, 0])+'.jpg')
        # print(img_path)
        
        # converts the image into a tensor 
        # use PIL to open the image
        image = Image.open(img_path)
        # convert image to RGB, we have single channel images
        image = image.convert("RGB")

        label = self.img_labels.iloc[idx, 1:].values
        if self.transform: #apply transform to image
            image = self.transform(image)
        if self.target_transform: #apply transform to target
            label = self.target_transform(label)
        return image, label
