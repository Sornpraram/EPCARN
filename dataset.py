import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from skimage import img_as_float
from random import randrange

import sys
import numpy as np
import pyzed.sl as sl
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, scale):

    target = modcrop(Image.open(join(filepath)).convert('RGB'), scale)
    #print(type(target))
    input = target   #Streaming_LR
    #input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)  #Original
    #print(type(input))
    input = np.array(target)

    preset_img_quality = 1
    img_send_jpeg = cv2.imencode(".jpg", input, [cv2.IMWRITE_JPEG_QUALITY, preset_img_quality])[1]
    nparr = np.asarray(img_send_jpeg, np.uint8)
    img_recv_jpeg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    PIL_image = Image.fromarray(img_recv_jpeg.astype('uint8'), 'RGB')


    #print('type target: ',type(target))
    #print('type input: ',type(img_recv_jpeg))
    #print('type pil input: ',type(PIL_image))
    #img_recv_jpeg = img_recv_jpeg.convert('RGB')

    return target, PIL_image

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo)
    iw = iw - (iw%modulo)
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, patch_size, scale, nFrames, patch_count):
    #print("img_in", img_in.size[0])
    #print("img_tar", img_tar.size[0])
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    patch_count += 1

    if patch_count <= 115788:
        ix = 85 #random.randrange(0, iw - ip + 1)
        iy = 85 #random.randrange(0, ih - ip + 1)


    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_tar.crop((ty,tx,ty + tp, tx + tp))#img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))#[:, ty:ty + tp, tx:tx + tp]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}
    
    #print(patch_size)
    #print(info_patch)
    #print("img_in", img_in.size[0])
    #print("img_tar", img_tar.size[0])

    return img_in, img_tar, info_patch


    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def resize_im(input_in, scale):
    '''
    img_in = img_in.resize((int(img_in.size[0]/scale),int(img_in.size[1]/scale)))
    #img_in = np.array(img_in)
    #print(type(img_in))
    lr_show = img_in.resize((img_in.size[0],img_in.size[1]))
    img_in = np.array(img_in)
    lr_show = np.array(lr_show)

    img_send_jpeg = cv2.imencode(".jpg", img_in, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    nparr = np.asarray(img_send_jpeg, np.uint8)
    img_recv_jpeg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    lr_show = cv2.imencode(".jpg", lr_show, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
    lr_show = np.asarray(lr_show, np.uint8)
    lr_show = cv2.imdecode(lr_show, cv2.IMREAD_COLOR)
    '''
    #print(input_in.size[0])
    input_out = input_in.resize((int(input_in.size[0]/scale),int(input_in.size[1]/scale)))
    #input_out_show = input_out
    #input_out = input_out.resize((int(128),int(128)))
    input_out_show = input_out.resize((256,256))


    return input_out, input_out_show

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,nFrames, upscale_factor, data_augmentation, other_dataset, patch_size, future_frame, transform=None):
        super(DatasetFromFolder, self).__init__()
        files = [files for files in listdir(image_dir)]
        self.image_filenames = [join(image_dir,x) for x in files]
        #print(self.image_filenames)
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame

        self.patch_count = 1

    def __getitem__(self, index):
        #print("Current Index: ", self.image_filenames[index])
        
        target, input = load_img(self.image_filenames[index], self.upscale_factor)

        if self.patch_size != 0:
            input, target, _ = get_patch(input, target, self.patch_size, self.upscale_factor, self.nFrames, self.patch_count)
            
        #bicubic = rescale_img(input, self.upscale_factor) ##original
        LR, lr_show = resize_im(input, scale=2) ##streaming LR
    

        if self.transform:
            target = self.transform(target)
            #input = self.transform(input) #Original
            input = self.transform(LR) #Streaming_LR

            bicubic = self.transform(lr_show)

        return input, target, bicubic

    def __len__(self):
        return len(self.image_filenames)

'''
class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DatasetFromFolderTest, self).__init__()
        files = [files for files in listdir(image_dir)]
        self.image_filenames = [join(image_dir,x) for x in files]
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame

    def __getitem__(self, index):
        
        target, input = load_img(self.image_filenames[index], self.upscale_factor)
            
        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            
        return input, target, bicubic
      
    def __len__(self):
        return len(self.image_filenames)

'''
'''
def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug
'''