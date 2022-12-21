from keras.utils import to_categorical

import numpy as np
import keras
import cv2
import glob
import os

class DataGen(keras.utils.Sequence):
    def __init__(self, image_path, label_path, batch_size=16, image_size=128):
        self.image_path = image_path
        self.label_path = label_path
        self.train_files=glob.glob(image_path + '*.png')
        self.label_files=glob.glob(label_path + '*.png')
        self.batch_size = batch_size
        self.image_size = image_size
        self.steps = len(self.train_files)//batch_size
        self.on_epoch_end()
        
    def __load__(self, file_name):        

        ## Reading Image
        image = cv2.imread(file_name, 1)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image).astype(np.uint8)/255.0

        ## Reading Label
        head,file_name=os.path.split(file_name)
        head,_=os.path.split(head)
        head,seq_name=os.path.split(head)
        label_file=glob.glob(os.path.join(os.path.join(os.path.join(self.label_path,seq_name),"TrainId"),file_name))[0]
        label = cv2.imread(label_file, 1)
        label = cv2.resize(label, (self.image_size, self.image_size))
        label = to_categorical(label[:,:,0],num_classes=8)

        return image, label

    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.train_files):
            self.batch_size = len(self.train_files) - index*self.batch_size
        
        files_batch = self.train_files[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.train_files)/float(self.batch_size)))