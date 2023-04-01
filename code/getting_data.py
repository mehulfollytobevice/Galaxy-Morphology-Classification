from zipfile import ZipFile
import shutil
from pathlib import Path
import re
import os
import pandas as pd
import random
import sys
random.seed(42)

# what is the path of our file 
PATH="YOUR PATH"

# try removing prior train/test folders
try:
    shutil.rmtree(PATH+'/data/train')
    shutil.rmtree(PATH+'/data/test')
    print('Removing already existing folders')
except:
    print('Train/test not there')

# let's extract images
galaxy_zip = ZipFile(PATH+'/data/images_training_rev1.zip','r')
files=galaxy_zip.namelist()
print('Total number of files:',len(files))
n=int(sys.argv[1]) #number of images to extract

# random selection of files
random.shuffle(files)
files=files[:n]

# extract the images
for i in files:
    galaxy_zip.extract(i)
    print('Extracting image:',i)
    
galaxy_zip.close()

# make folders to store the data
os.mkdir(PATH+'/data/train')
os.mkdir(PATH+'/data/test')

# split into train and test
train=files[:int(n*.80)]
test= files[int(n*.80):]

print('--'*50)
# move training files
for fi in train:
    try:
        fi=fi.split('/')[-1]
        shutil.move(PATH+f"/code/images_training_rev1/{fi}", PATH+f"/data/train/{fi}",copy_function = shutil.copy2)
        print('Image moved to train:',fi)
    except OSError as e:
        print('Some problem moving train image',fi,e)
    

# move testing files
for fi in test:
    try:
        fi=fi.split('/')[-1]
        shutil.move(PATH+f'/code/images_training_rev1/{fi}', PATH+f'/data/test/{fi}',copy_function = shutil.copy2)
        print('Image moved to test:',fi)
    except OSError as e:
        print('Some problem moving test image',fi)

# remove the empty folder
os.rmdir(PATH+'/code/images_training_rev1')

path=Path(PATH)
Path.BASE_path= path
fname = Path(path/"data/train").glob('./*')

# find all the file names
train_idx=[int(re.findall(r'(\d+).jpg$',f)[0]) for f in train]
test_idx=[int(re.findall(r'(\d+).jpg$',f)[0]) for f in test]

df=pd.read_csv(PATH+'/data/training_solutions_rev1.csv')

train_df= df.loc[df['GalaxyID'].isin(train_idx)]
test_df= df.loc[df['GalaxyID'].isin(test_idx)]

# save the train/test csv files
print('Saving csv files')
train_df.to_csv(path/'data/train.csv',index=False)
test_df.to_csv(path/'data/test.csv',index=False)
