#from scipy.misc import imread, imsave
import torch
from torch.autograd import Variable
from canny.net_canny import Net


def canny(raw_img, use_cuda=False):
    data = raw_img #torch.from_numpy(raw_img.transpose((2, 0, 1)))
    #batch = torch.stack([img]).float()
    #return batch.view(1,3,batch.shape[-2],batch.shape[-1])
    net = Net(threshold=1.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()
    #torch.autograd.set_detect_anomaly(True)

    

    thresholded = net(data)
    return thresholded

    '''imsave('gradient_magnitude.png',grad_mag.data.cpu().numpy()[0,0])
    imsave('thin_edges.png', thresholded.data.cpu().numpy()[0, 0])
    imsave('final.png', (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))
    imsave('thresholded.png', early_threshold.data.cpu().numpy()[0, 0])'''


if __name__ == '__main__':
    img = imread('fb_profile.jpg') / 255.0

    # canny(img, use_cuda=False)
    canny(img, use_cuda=True)
