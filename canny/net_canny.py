import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian
import time
import sys


class Net(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=False,size=[360,640]): #train 256,256 #test 360,640
        super(Net, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda
        filter_size = 3
        #generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        '''self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))'''

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_horizontal = self.sobel_filter_horizontal.half()



        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = self.sobel_filter_vertical.half()

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))
        self.directional_filter = self.directional_filter.half()

        #pixel_range = torch.FloatTensor([range(pixel_count)])
        self.pixel_range = torch.cuda.FloatTensor([range(size[0]*size[1])])
        self.size = size

    def forward(self, img):
        img = (img[:,0:1] + img[:,1:2] + img[:,2:3])/3.0
        #blur_horizontal = self.gaussian_filter_horizontal(img)
        #blurred_img = self.gaussian_filter_vertical(blur_horizontal)

        grad_x = self.sobel_filter_horizontal(img)
        
        grad_y = self.sobel_filter_vertical(img)
        # COMPUTE THICK EDGES
        
        grad_mag = torch.sqrt_(grad_x**2 + grad_y**2)
        
        grad_orientation = (torch.atan2(grad_y, grad_x) * (180.0/3.14159)) + 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0
        # grad_orientation =  torch.round( grad_orientatpixel_countion / 45.0 ) * 45.0
        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8
        pixel_count = self.size[0]*self.size[1]

        inidices_positive =inidices_positive.float()
        inidices_negative = inidices_negative.float()
        all_filtered = all_filtered.float()
        #print(inidices_positive.size())
        #print(pixel_count)
        #print(self.pixel_range.size())
        
        indices = (inidices_positive.view(-1).data * pixel_count + self.pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,self.size[0],self.size[1])

        indices = (inidices_negative.view(-1).data * pixel_count + self.pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,self.size[0],self.size[1])

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])
        
        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)
        thin_edges = grad_mag#.contiguous()
        thin_edges[is_max==0.0] = 0.0

        #thresholded = thin_edges.contiguous()
        #thresholded[thin_edges<0.5] = 0.0

        early_threshold = grad_mag#.contiguous()
        early_threshold[grad_mag<0.2] = 0.0
        
        '''grad_mag[is_max==0] = 0.0
        
        # THRESHOLD
        #thresholded = thin_edges.contiguous()
        #thresholded[thin_edges<0.0*self.threshold] = 0.0
        print('6',time.time()-t)'''
        th = early_threshold
        th[th > 0.0] = 1
        th[th <= 0.0] = 0
        return th.detach()

        #early_threshold = grad_mag.contiguous()
        #early_threshold[grad_mag<self.threshold] = 0.0

        #assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()
        
        #return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold



if __name__ == '__main__':
    Net()
