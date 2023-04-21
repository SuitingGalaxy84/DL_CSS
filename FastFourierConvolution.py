import torch
import numpy as np
import torch.nn as nn
import math
import torch.fft as FFT


"this module implement Fast Fourier Convolution on input data, it has advantages over vanilla convolution and posesses plug-and-play fashion"



#Specifications:
#        [B, C, H, W]:
#        B => batch_size
#        C => num_channel
#        H => height_pixel
#        W => width_pixel


#Parameters:
#        x: input mini_batch
#        channel_split: hyper-parameter to split the channel

#Attention:
#        the size of output is consistent to the input size
#        it is suggested that initial channel number of the global branch can be moded by 8


#Reference：

#https://papers.nips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf



#P.S. 
#this piece of code is written by a year 2 student so that there might be mis-interpretaion over the some detials 
#one is expected to double-check the module 

        

    









def Hermitian(y_i, y_r):
    hermitian = torch.zeros(y_i.shape[0], y_i.shape[1], y_i.shape[2],(y_i.shape[3]-1)*2)

    hermitian = torch.complex(hermitian, hermitian)
    
    h1 = torch.complex(y_r,  y_i)

    hermitian[: h1.shape[0], : h1.shape[1],: h1.shape[2], : h1.shape[3]] = h1

    for h in range(hermitian.shape[2]):

        for w in range(h1.shape[3]-1):

            hermitian[:,:,-h, -w] = torch.conj(hermitian[:,:,h,w])

    return hermitian





class FastFourierConvolution(nn.Module):
    def __init__(self, x, channel_split):
        super(nn.Module, self)
        self.x = x
        self.num_channel = x.shape[1]
        self.H = x.shape[2]
        self.W = x.shape[3]
        self.x_g = x[:, 0: math.ceil((1-channel_split)*x.shape[1]), :, :]
        self.x_l = x[:, math.ceil((1-channel_split)*x.shape[1]): x.shape[1]+1, :, :]




    def FourierUnit(self, tensor_in):#hermitian property X[i, j] = torch.conj(X[-i, -j])

        y = FFT.fft2(tensor_in, dim = (-2, -1))

        y = y[:, :, :, : math.floor(tensor_in.shape[3]/2) + 1]

        y_r, y_i = torch.real(y), torch.imag(y)

        y = torch.cat([y_r, y_i], dim =1)

        bn = nn.BatchNorm2d(y.shape[1])
        
        relu = nn.ReLU()
        
        Conv = nn.Conv2d(y.shape[1], y.shape[1], 1)

        y = relu(bn(Conv(y)))

        y_r, y_i = torch.chunk(y, 2, dim=1)

        hermitian = Hermitian(y_i, y_r)       
        
        z = FFT.ifft2(hermitian).real
        return z
        

    
    def LocalFourierUnit(self, tensor_in):

        y5, y6= torch.chunk(tensor_in, 2, dim=2) #along the H

        y1, y2 = torch.chunk(y5, 2, 3) #along the W
        
        y3, y4 = torch.chunk(y6, 2, 3) #along the W

        y = torch.cat([y1, y2, y3, y4], dim=1)

        y = self.FourierUnit(y)

        y = torch.cat([y, y], dim=2)

        return torch.cat([y, y], dim=3)
    




    def LocalPath(self, tensor_in):

        conv2d_local = nn.Conv2d(tensor_in.shape[1], tensor_in.shape[1], kernel_size=1)

        return conv2d_local(tensor_in)
    



    
    def IntraPath(self, local_, global_):

        conv_l2g = nn.Conv2d(local_.shape[1], global_.shape[1], kernel_size= 1)

        conv_g2l = nn.Conv2d(global_.shape[1], local_.shape[1], kernel_size= 1)

        return conv_g2l(global_), conv_l2g(local_)
    



    def Transform(self, tensor_in):

        conv_in = nn.Conv2d(tensor_in.shape[1], math.floor(tensor_in.shape[1]/2), kernel_size=1)

        conv_out = nn.Conv2d(math.floor(tensor_in.shape[1]/2), tensor_in.shape[1], kernel_size=1)

        y_in = conv_in(tensor_in)

        y_g = self.FourierUnit(y_in)

        _, _, _, y_sg = torch.chunk(y_in, 4, dim=1)

        y_sg = self.LocalFourierUnit(y_sg)

        return conv_out((y_sg + y_g))
    



    def forward(self):

        Y_ll = self.LocalPath(self.x_l)
        
        Y_gg = self.Transform(self.x_g)
        
        g2l, l2g = self.IntraPath(self.x_l, self.x_g)
        
        Y_l = Y_ll + g2l
        
        Y_g = Y_gg + l2g

        out = torch.cat([Y_g, Y_l], dim=1)
        return out




x = np.array(list(range(4*16*256*256)),dtype=np.float32)
x = x.reshape([4, 16, 256, 256])
x = torch.from_numpy(x)
ffc = FastFourierConvolution(x, 0.5)
out = ffc.forward()







        