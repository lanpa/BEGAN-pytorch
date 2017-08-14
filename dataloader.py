import torchvision
import torchvision.transforms as tf
import glob
import os
import numpy as np
from PIL import Image
class ImageFolderSR():
    def __len__(self):
        return len(self.filenames)
    def __init__(self, root, HRsize, is_crop=False):
        self.root = root
        self.HRsize = HRsize
        self.is_crop = is_crop
        self.filenames = [os.path.basename(f) for f in glob.glob(os.path.join(root,'*.jpg'))+glob.glob(os.path.join(root,'*.png'))]
        print(root)
    def __getitem__(self, index):
        filename = self.filenames[index]
        img1x = Image.open(os.path.join(self.root, filename))#.crop((x*12,y*12, x*12+360, y*12+360))
        #img1x = Image.open(os.path.join(self.root, filename)).crop((25, 50, 25 + 128, 50 + 128)).resize((64, 64), Image.ANTIALIAS)

        if self.is_crop:
            #print(filename)
            #hr = tf.Scale([img1x.size[0]//4, img1x.size[1]//4])(img1x)                    
            hr = tf.RandomCrop([self.HRsize, self.HRsize])(img1x)        
        else:         
            
            hr = tf.Scale(self.HRsize)(img1x)# self.HRsize])(img1x)        
        #lr = tf.Scale([self.LRsize, self.LRsize])(img1x)        
        
        return [tf.ToTensor()(hr)*2-1]#, tf.ToTensor()(lr), tf.ToTensor()(hr)]
