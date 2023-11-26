import nibabel as nib
import numpy as np
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def generateDatasetPaths(path_to_data,path_training,path_validation):
    '''
    Generate lists containing folder names for training, tests and validation.
    In dataset we already have split between training and validation, 
    we will split validation dataset into 2 pieces: validation and testing.
    Afterwards we are going to shuffle them to gain random order
    '''
    path_training = path_to_data +'/'+ path_training
    path_validation = path_to_data +'/'+ path_validation

    train  =  pd.read_csv(path_training + '/name_mapping.csv')
    train = train[['BraTS_2020_subject_ID']]
    train['BraTS_2020_subject_ID'] = train['BraTS_2020_subject_ID'].apply(lambda x: path_training + '/' + x)
    train = train.sample(frac=1).reset_index(drop=True)

    validandtest  =  pd.read_csv(path_validation+'/name_mapping_validation_data.csv')
    validandtest = validandtest[['BraTS_2020_subject_ID']]
    validandtest['BraTS_2020_subject_ID'] = validandtest['BraTS_2020_subject_ID'].apply(lambda x: path_training + '/' + x)

    valid, test = train_test_split(validandtest, test_size=0.5)
    valid.reset_index(drop=True,inplace=True)
    test.reset_index(drop=True,inplace=True)

    print(f'Dataset split into datasets of size: \nTrain: {len(train)},\nTest: {len(test)},\nValid: {len(valid)}.')

    return train,test,valid

class BratsDataset(Dataset):
    def __init__(self, df_with_paths,batch_size = 8, channel = 0, slices_range = [30,130]):

        self.df_with_paths = df_with_paths
        self.batch_size = batch_size
        self.channel = channel
        self.slices_range = slices_range
    def __len__(self):
        return len(self.df_with_paths)

    def __getitem__(self, idx):
        filename  =  self.df_with_paths['BraTS_2020_subject_ID'][idx]
        # read 3D imagw
        print(os.path.join(filename,os.listdir(filename)[self.channel]))
        img3d = np.array(nib.load(os.path.join(filename,os.listdir(filename)[self.channel])).dataobj)
        label3d = np.array(nib.load(os.path.join(filename,os.listdir(filename)[1])).dataobj)

        
        img2d,label2d = self._process(img3d,label3d)
        return img2d,label2d

    def _process(self,img_3d,label_3d):
        '''
        Take image 3D and label 3D and return random slice from range "slices range"

        Optional:
        1. Data augmentation
        2. normalization
        3. other transformations
        '''
        #generate random z coordinate for slice
        z = np.random.randint(self.slices_range[0],self.slices_range[1])  
        # return slices
        return img_3d[:,:,z].astype(np.float32), label_3d[:,:,z].astype(np.float32)