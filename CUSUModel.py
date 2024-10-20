import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# import scipy
import torchvision

import numpy as np
from utils import awgn_channel, reparameterize
import gc

class Distributed_MultiTaskNetwork(nn.Module):
    def __init__(self, n_in, n_c_latent, n_c_h, n_x_in, n_x_latent, n_x_h,n_hidden_layer,conv_layer=True,transmit_CU=False,k_1=4, k_2=3, k_3=1,centralised_observation=False,SNR_task_1=0,SNR_task_2 = 0, additional_maxpool=False,pad=1,SNR_train_range=0):
        super(Distributed_MultiTaskNetwork, self).__init__()

        self.n_in = n_in
        self.n_c_latent = n_c_latent
        self.n_h = n_c_h
        self.n_x_in = n_x_in
        self.n_x_latent = n_x_latent
        self.n_x_h = n_x_h
        self.n_hidden_layer = n_hidden_layer
        self.conv_layer = conv_layer
        self.transmit_CU = transmit_CU
        self.centralised_observation = centralised_observation
        self.additional_maxpool = additional_maxpool

        self.pad = pad

        self.SNR_task_1 = SNR_task_1
        self.SNR_task_2 = SNR_task_2

        self.SNR_train_range=SNR_train_range
 
        if transmit_CU == True:
            self.n_x_latent = n_x_latent * 2
            n_x_latent = self.n_x_latent

        self.k_1 = k_1 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.k_2 = k_2
        self.k_3 = k_3

        self.transmit_CU_power_factor = 1 # can be set to 2 to make broadcast double the power


        if n_c_h == 0:
            n_c_h = n_in

        if n_hidden_layer == 0:
            n_hidden_layer = n_x_h

        if self.conv_layer == True:
            self.n_hidden_layer = k_3*3*3
            if self.centralised_observation == True:
                self.n_hidden_layer = k_3 * 7 * 7
                if self.pad == 0:
                    self.n_hidden_layer = k_3 * 5 * 5

                if self.additional_maxpool == True:
                    self.n_hidden_layer = k_3 * 3 * 3
            n_hidden_layer = self.n_hidden_layer

            self.n_h = 1
             #------ CU-Encoder agent 1
            self.cu_agent_1 = nn.Conv2d(1, k_1, kernel_size=(3,3), stride=1, padding=self.pad) #    n_in, n_c_h), nn.Tanh(),)
            self.relu_1_agent_1 = nn.ReLU()
            self.cu_output_mu_agent_1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_2_agent_1 = nn.Conv2d(k_1, k_2, kernel_size=(3,3), stride=1, padding=self.pad)
            self.relu_2_agent_1 = nn.ReLU()
            self.maxpool2_agent_1 = nn.MaxPool2d(kernel_size=(2, 2))
            # self.cu_output_ln_var_agent_1 = nn.Linear(n_c_h, n_c_latent)
    #------ CU-Encoder agent 2
            self.cu_agent_2 = nn.Conv2d(1, k_1, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_1_agent_2 = nn.ReLU()
            self.cu_output_mu_agent_2 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_2_agent_2 = nn.Conv2d(k_1, k_2, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_2_agent_2 = nn.ReLU()
            self.maxpool2_agent_2 = nn.MaxPool2d(kernel_size=(2, 2))
            # self.cu_output_ln_var_agent_2 = nn.Linear(n_c_h, n_c_latent) 
    #------ CU-Encoder agent 3
            self.cu_agent_3 = nn.Conv2d(1, k_1, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_1_agent_3 = nn.ReLU()
            self.cu_output_mu_agent_3 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_2_agent_3 = nn.Conv2d(k_1, k_2, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_2_agent_3 = nn.ReLU()
            self.maxpool2_agent_3 = nn.MaxPool2d(kernel_size=(2, 2))
            # self.cu_output_ln_var_agent_3 = nn.Linear(n_c_h, n_c_latent)
    #------ CU-Encoder agent 4
            self.cu_agent_4 = nn.Conv2d(1, k_1, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_1_agent_4 = nn.ReLU()
            self.cu_output_mu_agent_4 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_2_agent_4 = nn.Conv2d(k_1, k_2, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_2_agent_4 = nn.ReLU()
            self.maxpool2_agent_4 = nn.MaxPool2d(kernel_size=(2, 2))
            # self.cu_output_ln_var_agent_4 = nn.Linear(n_c_h, n_c_latent)

    #------ SU1-Encoder agent 1
            self.sue_one_agent_1 = nn.Conv2d(k_2, k_3, kernel_size=(3,3), stride=1, padding=1)
            self.su_relu_1_agent_1 = nn.ReLU()
            self.su_one_hidden_layer_agent_1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.sue_one_output_agent_1 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU1-Encoder agent 2
            self.sue_one_agent_2 = nn.Conv2d(k_2, k_3, kernel_size=(3, 3), stride=1, padding=1)
            self.su_relu_1_agent_2 = nn.ReLU()
            self.su_one_hidden_layer_agent_2 = nn.MaxPool2d(kernel_size=(2, 2))
            self.sue_one_output_agent_2 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU1-Encoder agent 3
            self.sue_one_agent_3 = nn.Conv2d(k_2, k_3, kernel_size=(3, 3), stride=1, padding=1)
            self.su_relu_1_agent_3 = nn.ReLU()
            self.su_one_hidden_layer_agent_3 = nn.MaxPool2d(kernel_size=(2, 2))
            self.sue_one_output_agent_3 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU1-Encoder agent 4
            self.sue_one_agent_4 = nn.Conv2d(k_2, k_3, kernel_size=(3, 3), stride=1, padding=1)
            self.su_relu_1_agent_4 = nn.ReLU()
            self.su_one_hidden_layer_agent_4 = nn.MaxPool2d(kernel_size=(2, 2))
            self.sue_one_output_agent_4 = nn.Linear(n_hidden_layer, n_x_latent)

            # SU1-Decoder  COMBINED
            if self.centralised_observation == True:
                n_x_latent_decode = n_x_latent
            else:
                n_x_latent_decode = n_x_latent*4

            self.sud_one = nn.Sequential(nn.Linear(n_x_latent_decode, 16), nn.Tanh(),)
            self.sud_one_out = nn.Linear(16, 1)

    #------ SU2-Encoder agent 1
            self.sue_two_agent_1 = nn.Conv2d(k_2, k_3, kernel_size=(3,3), stride=1, padding=1)
            self.su_two_relu_1_agent_1 = nn.ReLU()
            self.su_two_hidden_layer_agent_1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.sue_two_output_agent_1 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU2-Encoder agent 2
            self.sue_two_agent_2 = nn.Conv2d(k_2, k_3, kernel_size=(3, 3), stride=1, padding=1)
            self.su_two_relu_1_agent_2 = nn.ReLU()
            self.su_two_hidden_layer_agent_2 = nn.MaxPool2d(kernel_size=(2, 2))
            self.sue_two_output_agent_2 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU2-Encoder agent 3
            self.sue_two_agent_3 = nn.Conv2d(k_2, k_3, kernel_size=(3, 3), stride=1, padding=1)
            self.su_two_relu_1_agent_3 = nn.ReLU()
            self.su_two_hidden_layer_agent_3 = nn.MaxPool2d(kernel_size=(2, 2))
            self.sue_two_output_agent_3 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU2-Encoder agent 4
            self.sue_two_agent_4 = nn.Conv2d(k_2, k_3, kernel_size=(3, 3), stride=1, padding=1)
            self.su_two_relu_1_agent_4 = nn.ReLU()
            self.su_two_hidden_layer_agent_4 = nn.MaxPool2d(kernel_size=(2, 2))
            self.sue_two_output_agent_4 = nn.Linear(n_hidden_layer, n_x_latent)

            # SU2-Decoder COMBINED
            if self.centralised_observation == True:
                n_x_latent_decode = n_x_latent
            else:
                n_x_latent_decode = n_x_latent*4

            self.sud_two = nn.Sequential(nn.Linear(n_x_latent_decode, 16), nn.Tanh(),)
            self.sud_two_out = nn.Linear(16, 10) 
        else:
    #------ CU-Encoder agent 1
            self.cu_agent_1 = nn.Sequential(nn.Linear(n_in, n_c_h), nn.Tanh(),)
            self.cu_output_mu_agent_1 = nn.Sequential(nn.Linear(n_c_h, n_c_latent), nn.Tanh(),)
            # self.cu_output_ln_var_agent_1 = nn.Linear(n_c_h, n_c_latent)
    #------ CU-Encoder agent 2
            self.cu_agent_2 = nn.Sequential(nn.Linear(n_in, n_c_h), nn.Tanh(),)
            self.cu_output_mu_agent_2 = nn.Sequential(nn.Linear(n_c_h, n_c_latent), nn.Tanh(),)
            # self.cu_output_ln_var_agent_2 = nn.Linear(n_c_h, n_c_latent) 
    #------ CU-Encoder agent 3
            self.cu_agent_3 = nn.Sequential(nn.Linear(n_in, n_c_h), nn.Tanh(),)
            self.cu_output_mu_agent_3 = nn.Sequential(nn.Linear(n_c_h, n_c_latent), nn.Tanh(),)
            # self.cu_output_ln_var_agent_3 = nn.Linear(n_c_h, n_c_latent)
    #------ CU-Encoder agent 4
            self.cu_agent_4 = nn.Sequential(nn.Linear(n_in, n_c_h), nn.Tanh(),)
            self.cu_output_mu_agent_4 = nn.Sequential(nn.Linear(n_c_h, n_c_latent), nn.Tanh(),)
            # self.cu_output_ln_var_agent_4 = nn.Linear(n_c_h, n_c_latent)

    #------ SU1-Encoder agent 1
            self.sue_one_agent_1 = nn.Sequential(nn.Linear(n_x_in, n_x_h), nn.Tanh(),)
            self.su_one_hidden_layer_agent_1 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.sue_one_output_agent_1 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU1-Encoder agent 2
            self.sue_one_agent_2 = nn.Sequential(nn.Linear(n_x_in, n_x_h), nn.Tanh(),)
            self.su_one_hidden_layer_agent_2 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.sue_one_output_agent_2 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU1-Encoder agent 3
            self.sue_one_agent_3 = nn.Sequential(nn.Linear(n_x_in, n_x_h), nn.Tanh(),)
            self.su_one_hidden_layer_agent_3 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.sue_one_output_agent_3 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU1-Encoder agent 4
            self.sue_one_agent_4 = nn.Sequential(nn.Linear(n_x_in, n_x_h), nn.Tanh(),)
            self.su_one_hidden_layer_agent_4 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.sue_one_output_agent_4 = nn.Linear(n_hidden_layer, n_x_latent)

            # SU1-Decoder  COMBINED
            if self.centralised_observation == True:
                n_x_latent_decode = n_x_latent
            else:
                n_x_latent_decode = n_x_latent*4

            self.sud_one = nn.Sequential(nn.Linear(n_x_latent_decode, 16), nn.Tanh(),)
            self.sud_one_out = nn.Linear(16, 1)

    #------ SU2-Encoder agent 1
            self.sue_two_agent_1 = nn.Sequential(nn.Linear(n_x_in, n_x_h), nn.Tanh(),)
            self.su_two_hidden_layer_agent_1 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.sue_two_output_agent_1 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU2-Encoder agent 2
            self.sue_two_agent_2 = nn.Sequential(nn.Linear(n_x_in, n_x_h), nn.Tanh(),)
            self.su_two_hidden_layer_agent_2 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.sue_two_output_agent_2 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU2-Encoder agent 3
            self.sue_two_agent_3 = nn.Sequential(nn.Linear(n_x_in, n_x_h), nn.Tanh(),)
            self.su_two_hidden_layer_agent_3 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.sue_two_output_agent_3 = nn.Linear(n_hidden_layer, n_x_latent)
    #------ SU2-Encoder agent 4
            self.sue_two_agent_4 = nn.Sequential(nn.Linear(n_x_in, n_x_h), nn.Tanh(),)
            self.su_two_hidden_layer_agent_4 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.sue_two_output_agent_4 = nn.Linear(n_hidden_layer, n_x_latent)

            # SU2-Decoder COMBINED
            if self.centralised_observation == True:
                n_x_latent_decode = n_x_latent
            else:
                n_x_latent_decode = n_x_latent*4

            self.sud_two = nn.Sequential(nn.Linear(n_x_latent_decode, 16), nn.Tanh(),)
            self.sud_two_out = nn.Linear(16, 10) 

        
    def cu_encode_agent_1(self, s, task=1):
        if self.conv_layer == True:
            self.batch_size = s.shape[0]
            if self.centralised_observation == True:
                h = s.reshape((self.batch_size,1,28,28))
            else:
                h = s.reshape((self.batch_size,1,14, 14))
            h = self.cu_agent_1(h)
            h = self.relu_1_agent_1(h)
            h = self.cu_output_mu_agent_1(h)
            h = self.conv_2_agent_1(h)
            h = self.relu_2_agent_1(h)
            h = self.maxpool2_agent_1(h)
            c = h
            # return c
            if self.transmit_CU == True:
                h = self.sue_one_agent_1(c)
                h = self.su_relu_1_agent_1(h)
                # h = self.su_one_hidden_layer_agent_1(h)
                h = h.reshape((self.batch_size,self.n_hidden_layer))
                return np.sqrt(self.batch_size*self.transmit_CU_power_factor)*torch.nn.functional.normalize(self.sue_one_output_agent_1(h),dim=(0,1))
        else:    
            if self.n_h > 0:
                h = self.cu_agent_1(s)
            else:
                h = s
            self.batch_size = h.shape[0]
            return self.cu_output_mu_agent_1(h) #, self.cu_output_ln_var_agent_1(h)   
    #def su_t1_encode_agent_1(self, c):
        if self.conv_layer == True and task==1:
            h = self.sue_one_agent_1(c)
            h = self.su_relu_1_agent_1(h)
            if self.additional_maxpool == True:
                h = self.su_one_hidden_layer_agent_1(h)
            h = h.reshape((self.batch_size,self.n_hidden_layer))
            if self.centralised_observation == True: #same total power for central and distributed case
                return np.sqrt(self.batch_size*4)*torch.nn.functional.normalize(self.sue_one_output_agent_1(h),dim=(0,1))
            return np.sqrt(self.batch_size)*torch.nn.functional.normalize(self.sue_one_output_agent_1(h),dim=(0,1))
        elif task==1:
            h = self.sue_one_agent_1(c)
            if self.n_hidden_layer > 0:
                h = self.su_one_hidden_layer_agent_1(h)
            return np.sqrt(self.batch_size)*torch.nn.functional.normalize(self.sue_one_output_agent_1(h),dim=(0,1)) #normalize output to power of 1, averaged over latent dim and batches
    # def su_t2_encode_agent_1(self, c):
        if self.conv_layer == True:
            h = self.sue_two_agent_1(c)
            h = self.su_two_relu_1_agent_1(h)
            if self.additional_maxpool == True:
                h = self.su_two_hidden_layer_agent_1(h)
            h = h.reshape((self.batch_size,self.n_hidden_layer))
            if self.centralised_observation == True: #same total power for central and distributed case
                return np.sqrt(self.batch_size*4)*torch.nn.functional.normalize(self.sue_two_output_agent_1(h),dim=(0,1))
            return np.sqrt(self.batch_size)*torch.nn.functional.normalize(self.sue_two_output_agent_1(h),dim=(0,1))
        else:
            h = self.sue_two_agent_1(c)
            if self.n_hidden_layer > 0:
                h = self.su_two_hidden_layer_agent_1(h)
            return np.sqrt(self.batch_size)*torch.nn.functional.normalize(self.sue_two_output_agent_1(h),dim=(0,1)) #normalize output

    def cu_encode_agent_2(self, s, task=1):
        if self.conv_layer == True:
            h = s.reshape((self.batch_size, 1, 14, 14))
            h = self.cu_agent_2(h)
            h = self.relu_1_agent_2(h)
            h = self.cu_output_mu_agent_2(h)
            h = self.conv_2_agent_2(h)
            h = self.relu_2_agent_2(h)
            h = self.maxpool2_agent_2(h)
            c = h
            # return c
            if self.transmit_CU == True:
                h = self.sue_one_agent_2(c)
                h = self.su_relu_1_agent_2(h)
                # h = self.su_one_hidden_layer_agent_1(h)
                h = h.reshape((self.batch_size,self.n_hidden_layer))
                return np.sqrt(self.batch_size*self.transmit_CU_power_factor)*torch.nn.functional.normalize(self.sue_one_output_agent_2(h),dim=(0,1))
        else:    
            if self.n_h > 0:
                h = self.cu_agent_2(s)
            else:
                h = s
            self.batch_size = h.shape[0]
            return self.cu_output_mu_agent_2(h)
    # def su_t1_encode_agent_2(self, c):
        if self.conv_layer == True and task==1:
            h = self.sue_one_agent_2(c)
            h = self.su_relu_1_agent_2(h)
            if self.additional_maxpool == True:
                h = self.su_one_hidden_layer_agent_2(h)
            h = h.reshape((self.batch_size, self.n_hidden_layer))
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_one_output_agent_2(h), dim=(0, 1))
        elif task==1:
            h = self.sue_one_agent_2(c)
            if self.n_hidden_layer > 0:
                h = self.su_one_hidden_layer_agent_2(h)
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_one_output_agent_2(h), dim=(0, 1))
    # def su_t2_encode_agent_2(self, c):
        if self.conv_layer == True:
            h = self.sue_two_agent_2(c)
            h = self.su_two_relu_1_agent_2(h)
            if self.additional_maxpool == True:
                h = self.su_two_hidden_layer_agent_2(h)
            h = h.reshape((self.batch_size, self.n_hidden_layer))
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_two_output_agent_2(h), dim=(0, 1))
        else:
            h = self.sue_two_agent_2(c)
            if self.n_hidden_layer > 0:
                h = self.su_two_hidden_layer_agent_2(h)
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_two_output_agent_2(h), dim=(0, 1))
        
    def cu_encode_agent_3(self, s, task=1):
        if self.conv_layer == True:
            h = s.reshape((self.batch_size, 1, 14, 14))
            h = self.cu_agent_3(h)
            h = self.relu_1_agent_3(h)
            h = self.cu_output_mu_agent_3(h)
            h = self.conv_2_agent_3(h)
            h = self.relu_2_agent_3(h)
            h = self.maxpool2_agent_3(h)
            c = h
            # return c
            if self.transmit_CU == True:
                h = self.sue_one_agent_3(c)
                h = self.su_relu_1_agent_3(h)
                # h = self.su_one_hidden_layer_agent_1(h)
                h = h.reshape((self.batch_size,self.n_hidden_layer))
                return np.sqrt(self.batch_size*self.transmit_CU_power_factor)*torch.nn.functional.normalize(self.sue_one_output_agent_3(h),dim=(0,1))
        else:    
            if self.n_h > 0:
                h = self.cu_agent_3(s)
            else:
                h = s
            self.batch_size = h.shape[0]
            return self.cu_output_mu_agent_3(h)
    # def su_t1_encode_agent_3(self, c):
        if self.conv_layer == True and task==1:
            h = self.sue_one_agent_3(c)
            h = self.su_relu_1_agent_3(h)
            if self.additional_maxpool == True:
                h = self.su_one_hidden_layer_agent_3(h)
            h = h.reshape((self.batch_size, self.n_hidden_layer))
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_one_output_agent_3(h), dim=(0, 1))
        elif task==1:
            h = self.sue_one_agent_3(c)
            if self.n_hidden_layer > 0:
                h = self.su_one_hidden_layer_agent_3(h)
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_one_output_agent_3(h), dim=(0, 1))
    # def su_t2_encode_agent_3(self, c):
        if self.conv_layer == True:
            h = self.sue_two_agent_3(c)
            h = self.su_two_relu_1_agent_3(h)
            if self.additional_maxpool == True:
                h = self.su_two_hidden_layer_agent_3(h)
            h = h.reshape((self.batch_size, self.n_hidden_layer))
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_two_output_agent_3(h), dim=(0, 1))
        else:
            h = self.sue_two_agent_3(c)
            if self.n_hidden_layer > 0:
                h = self.su_two_hidden_layer_agent_3(h)
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_two_output_agent_3(h), dim=(0, 1))

    def cu_encode_agent_4(self, s, task=1):
        if self.conv_layer == True:
            h = s.reshape((self.batch_size, 1, 14, 14))
            h = self.cu_agent_4(h)
            h = self.relu_1_agent_4(h)
            h = self.cu_output_mu_agent_4(h)
            h = self.conv_2_agent_4(h)
            h = self.relu_2_agent_4(h)
            h = self.maxpool2_agent_4(h)
            c = h
            # return c
            if self.transmit_CU == True:
                h = self.sue_one_agent_4(c)
                h = self.su_relu_1_agent_4(h)
                # h = self.su_one_hidden_layer_agent_1(h)
                h = h.reshape((self.batch_size,self.n_hidden_layer))
                return np.sqrt(self.batch_size*self.transmit_CU_power_factor)*torch.nn.functional.normalize(self.sue_one_output_agent_4(h),dim=(0,1))
        else:    
            if self.n_h > 0:
                h = self.cu_agent_4(s)
            else:
                h = s
            self.batch_size = h.shape[0]
            return self.cu_output_mu_agent_4(h)
    # def su_t1_encode_agent_4(self, c):
        if self.conv_layer == True and task==1:
            h = self.sue_one_agent_4(c)
            h = self.su_relu_1_agent_4(h)
            if self.additional_maxpool == True:
                h = self.su_one_hidden_layer_agent_4(h)
            h = h.reshape((self.batch_size, self.n_hidden_layer))
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_one_output_agent_4(h), dim=(0, 1))
        elif task == 1:
            h = self.sue_one_agent_4(c)
            if self.n_hidden_layer > 0:
                h = self.su_one_hidden_layer_agent_4(h)
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_one_output_agent_4(h), dim=(0, 1))
    # def su_t2_encode_agent_4(self, c):
        if self.conv_layer == True:
            h = self.sue_two_agent_4(c)
            h = self.su_two_relu_1_agent_4(h)
            if self.additional_maxpool == True:
                h = self.su_two_hidden_layer_agent_4(h)
            h = h.reshape((self.batch_size, self.n_hidden_layer))
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_two_output_agent_4(h), dim=(0, 1))
        else:
            h = self.sue_two_agent_4(c)
            if self.n_hidden_layer > 0:
                h = self.su_two_hidden_layer_agent_4(h)
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.sue_two_output_agent_4(h), dim=(0, 1))

    def su_t1_decode(self, x):
        h = self.sud_one(x)
        return self.sud_one_out(h)

    def su_t2_decode(self, x):
        h = self.sud_two(x)
        return self.sud_two_out(h)


class Distributed_SU_ohne_CUnet(nn.Module):
    def __init__(self, n_x_h, n_x_latent,n_hidden_layer = 0, Nr_decoder_outputs=1,conv_layer=True,s_1=3, s_2=2, s_3=1,centralised_observation=False,SNR = 0, additional_maxpool = False, pad=1,SNR_train_range=0):
        super(Distributed_SU_ohne_CUnet, self).__init__()

        self.n_x_h = n_x_h
        self.n_hidden_layer = n_hidden_layer
        self.n_x_latent = n_x_latent
        self.conv_layer = conv_layer
        self.centralised_observation = centralised_observation
        self.additional_maxpool = additional_maxpool

        self.pad = pad
        self.SNR = SNR

        if n_hidden_layer == 0:
            n_hidden_layer = n_x_h

        self.s_1 = s_1 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.s_2 = s_2
        self.s_3 = s_3

        if self.conv_layer == True:
            if self.centralised_observation == True:
                self.n_hidden_layer = s_3 * 7 * 7
                if self.pad == 0:
                    self.n_hidden_layer = s_3 * 5 * 5
                if self.additional_maxpool == True:
                    self.n_hidden_layer = s_3 * 3 * 3
            else:
                self.n_hidden_layer = s_3*3*3
            n_hidden_layer = self.n_hidden_layer

            #Encoder agent 1
            self.conv_1_agent_1 = nn.Conv2d(1, s_1, kernel_size=(3,3), stride=1, padding=self.pad)
            self.relu_1_agent_1 = nn.ReLU()
            self.maxpool1_agent_1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_2_agent_1 = nn.Conv2d(s_1, s_2, kernel_size=(3,3), stride=1, padding=self.pad)
            self.relu_2_agent_1 = nn.ReLU()
            self.maxpool2_agent_1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_3_agent_1 = nn.Conv2d(s_2, s_3, kernel_size=(3,3), stride=1, padding=1)
            self.relu_3_agent_1 = nn.ReLU()
            self.maxpool3_agent_1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.su_ohnecu_enc_output_agent_1 = nn.Linear(n_hidden_layer, n_x_latent)

            #Encoder agent 2
            self.conv_1_agent_2 = nn.Conv2d(1, s_1, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_1_agent_2 = nn.ReLU()
            self.maxpool1_agent_2 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_2_agent_2 = nn.Conv2d(s_1, s_2, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_2_agent_2 = nn.ReLU()
            self.maxpool2_agent_2 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_3_agent_2 = nn.Conv2d(s_2, s_3, kernel_size=(3, 3), stride=1, padding=1)
            self.relu_3_agent_2 = nn.ReLU()
            self.maxpool3_agent_2 = nn.MaxPool2d(kernel_size=(2, 2))
            self.su_ohnecu_enc_output_agent_2 = nn.Linear(n_hidden_layer, n_x_latent)

            #Encoder agent 3
            self.conv_1_agent_3 = nn.Conv2d(1, s_1, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_1_agent_3 = nn.ReLU()
            self.maxpool1_agent_3 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_2_agent_3 = nn.Conv2d(s_1, s_2, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_2_agent_3 = nn.ReLU()
            self.maxpool2_agent_3 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_3_agent_3 = nn.Conv2d(s_2, s_3, kernel_size=(3, 3), stride=1, padding=1)
            self.relu_3_agent_3 = nn.ReLU()
            self.maxpool3_agent_3 = nn.MaxPool2d(kernel_size=(2, 2))
            self.su_ohnecu_enc_output_agent_3 = nn.Linear(n_hidden_layer, n_x_latent)

            #Encoder agent 4
            self.conv_1_agent_4 = nn.Conv2d(1, s_1, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_1_agent_4 = nn.ReLU()
            self.maxpool1_agent_4 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_2_agent_4 = nn.Conv2d(s_1, s_2, kernel_size=(3, 3), stride=1, padding=self.pad)
            self.relu_2_agent_4 = nn.ReLU()
            self.maxpool2_agent_4 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv_3_agent_4 = nn.Conv2d(s_2, s_3, kernel_size=(3, 3), stride=1, padding=1)
            self.relu_3_agent_4 = nn.ReLU()
            self.maxpool3_agent_4 = nn.MaxPool2d(kernel_size=(2, 2))
            self.su_ohnecu_enc_output_agent_4 = nn.Linear(n_hidden_layer, n_x_latent)

        else:
            #Encoder agent 1
            self.su_ohnecu_enc_agent_1 = nn.Sequential(nn.Linear(196, n_x_h), nn.Tanh(),)
            self.su_ohnecu_enc_hidden_layer_agent_1 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.su_ohnecu_enc_output_agent_1 = nn.Linear(n_hidden_layer, n_x_latent)

            #Encoder agent 2
            self.su_ohnecu_enc_agent_2 = nn.Sequential(nn.Linear(196, n_x_h), nn.Tanh(),)
            self.su_ohnecu_enc_hidden_layer_agent_2 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.su_ohnecu_enc_output_agent_2 = nn.Linear(n_hidden_layer, n_x_latent)

            #Encoder agent 3
            self.su_ohnecu_enc_agent_3 = nn.Sequential(nn.Linear(196, n_x_h), nn.Tanh(),)
            self.su_ohnecu_enc_hidden_layer_agent_3 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.su_ohnecu_enc_output_agent_3 = nn.Linear(n_hidden_layer, n_x_latent)

            #Encoder agent 4
            self.su_ohnecu_enc_agent_4 = nn.Sequential(nn.Linear(196, n_x_h), nn.Tanh(),)
            self.su_ohnecu_enc_hidden_layer_agent_4 = nn.Sequential(nn.Linear(n_x_h, n_hidden_layer), nn.Tanh(),)
            self.su_ohnecu_enc_output_agent_4 = nn.Linear(n_hidden_layer, n_x_latent)

        
        #Decoder Combined
        if self.centralised_observation == True:
            n_x_latent_decode = n_x_latent
        else:
            n_x_latent_decode = n_x_latent*4

        self.su_ohnecu_dec = nn.Sequential(nn.Linear(n_x_latent_decode, 16), nn.Tanh(),)
        self.su_ohnecu_dec_out = nn.Linear(16, Nr_decoder_outputs)


    def su_ohnecu_encode_agent_1(self, s):
        if self.conv_layer == True:
            self.batch_size = s.shape[0]
            if self.centralised_observation == True:
                h = s.reshape((self.batch_size,1,28,28))
            else:
                h = s.reshape((self.batch_size,1,14, 14))

            h = self.conv_1_agent_1(h)
            h = self.relu_1_agent_1(h)
            h = self.maxpool1_agent_1(h)
            h = self.conv_2_agent_1(h)
            h = self.relu_2_agent_1(h)
            h = self.maxpool2_agent_1(h)
            h = self.conv_3_agent_1(h)
            h = self.relu_3_agent_1(h)
            if self.additional_maxpool == True:
                h = self.maxpool3_agent_1(h)
            h = h.reshape((self.batch_size,self.n_hidden_layer))
            if self.centralised_observation == True: #same total power for central and distributed case
                return np.sqrt(self.batch_size*4)*torch.nn.functional.normalize(self.su_ohnecu_enc_output_agent_1(h),dim=(0,1)) #normalize output
            return np.sqrt(self.batch_size)*torch.nn.functional.normalize(self.su_ohnecu_enc_output_agent_1(h),dim=(0,1)) #normalize output
        else:
            h = self.su_ohnecu_enc_agent_1(s)
            self.batch_size = h.shape[0]
            if self.n_hidden_layer > 0:
                h = self.su_ohnecu_enc_hidden_layer_agent_1(h)
            return np.sqrt(self.batch_size)*torch.nn.functional.normalize(self.su_ohnecu_enc_output_agent_1(h),dim=(0,1)) #normalize output
        
    def su_ohnecu_encode_agent_2(self, s):
        if self.conv_layer == True:
            h = s.reshape((self.batch_size, 1, 14, 14))
            h = self.conv_1_agent_2(h)
            h = self.relu_1_agent_2(h)
            h = self.maxpool1_agent_2(h)
            h = self.conv_2_agent_2(h)
            h = self.relu_2_agent_2(h)
            h = self.maxpool2_agent_2(h)
            h = self.conv_3_agent_2(h)
            h = self.relu_3_agent_2(h)
            if self.additional_maxpool == True:
                h = self.maxpool3_agent_2(h)
            h = h.reshape((self.batch_size,self.n_hidden_layer))
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.su_ohnecu_enc_output_agent_2(h), dim=(0, 1)) #normalize output
        else:
            h = self.su_ohnecu_enc_agent_2(s)
            self.batch_size = h.shape[0]
            if self.n_hidden_layer > 0:
                h = self.su_ohnecu_enc_hidden_layer_agent_2(h)
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.su_ohnecu_enc_output_agent_2(h), dim=(0, 1)) #normalize output
        
    def su_ohnecu_encode_agent_3(self, s):
        if self.conv_layer == True:
            h = s.reshape((self.batch_size, 1, 14, 14))
            h = self.conv_1_agent_3(h)
            h = self.relu_1_agent_3(h)
            h = self.maxpool1_agent_3(h)
            h = self.conv_2_agent_3(h)
            h = self.relu_2_agent_3(h)
            h = self.maxpool2_agent_3(h)
            h = self.conv_3_agent_3(h)
            h = self.relu_3_agent_3(h)
            if self.additional_maxpool == True:
                h = self.maxpool3_agent_3(h)
            h = h.reshape((self.batch_size,self.n_hidden_layer))
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.su_ohnecu_enc_output_agent_3(h), dim=(0, 1)) #normalize output
        else:
            h = self.su_ohnecu_enc_agent_3(s)
            self.batch_size = h.shape[0]
            if self.n_hidden_layer > 0:
                h = self.su_ohnecu_enc_hidden_layer_agent_3(h)
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.su_ohnecu_enc_output_agent_3(h), dim=(0, 1)) #normalize output

    def su_ohnecu_encode_agent_4(self, s):
        if self.conv_layer == True:
            h = s.reshape((self.batch_size, 1, 14, 14))
            h = self.conv_1_agent_4(h)
            h = self.relu_1_agent_4(h)
            h = self.maxpool1_agent_4(h)
            h = self.conv_2_agent_4(h)
            h = self.relu_2_agent_4(h)
            h = self.maxpool2_agent_4(h)
            h = self.conv_3_agent_4(h)
            h = self.relu_3_agent_4(h)
            if self.additional_maxpool == True:
                h = self.maxpool3_agent_4(h)
            h = h.reshape((self.batch_size,self.n_hidden_layer))
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.su_ohnecu_enc_output_agent_4(h), dim=(0, 1)) #normalize output
        else:
            h = self.su_ohnecu_enc_agent_4(s)
            self.batch_size = h.shape[0]
            if self.n_hidden_layer > 0:
                h = self.su_ohnecu_enc_hidden_layer_agent_4(h)
            return np.sqrt(self.batch_size) * torch.nn.functional.normalize(self.su_ohnecu_enc_output_agent_4(h), dim=(0, 1)) #normalize output
                                                                      
    def su_ohnecu_decode(self, x):
        h = self.su_ohnecu_dec(x)
        return self.su_ohnecu_dec_out(h)







class MultiTaskNetwork(nn.Module):
    def __init__(self, n_in, n_c_latent, n_c_h, n_x_in, n_x_latent, n_x_h):
        super(MultiTaskNetwork, self).__init__()

        self.n_in = n_in
        self.n_c_latent = n_c_latent
        self.n_h = n_c_h
        self.n_x_in = n_x_in
        self.n_x_latent = n_x_latent
        self.n_x_h = n_x_h

#------ CU-Encoder
        self.cu = nn.Sequential(
            nn.Linear(n_in, n_c_h), nn.Tanh(),
        )
        self.cu_output_mu = nn.Linear(n_c_h, n_c_latent)
        self.cu_output_ln_var = nn.Linear(n_c_h, n_c_latent)

#------ SU1-Encoder
        self.sue_one = nn.Sequential(
            nn.Linear(n_x_in, n_x_h), nn.Tanh(),
        )
        self.sue_one_output = nn.Linear(n_x_h, n_x_latent)

        # SU1-Decoder
        self.sud_one = nn.Sequential(
            nn.Linear(n_x_latent, 16), nn.Tanh(),
        )
        self.sud_one_out = nn.Linear(16, 1)

#------ SU2-Encoder
        self.sue_two = nn.Sequential(
            nn.Linear(n_x_in, n_x_h), nn.Tanh(),
        )
        self.sue_two_output = nn.Linear(n_x_h, n_x_latent)

        # SU2-Decoder
        self.sud_two = nn.Sequential(
            nn.Linear(n_x_latent, 16), nn.Tanh(),
        )
        self.sud_two_out = nn.Linear(16, 10) 

    def cu_encode(self, s):
        h = self.cu(s)
        return self.cu_output_mu(h), self.cu_output_ln_var(h)
    
    def su_t1_encode(self, c):
        h = self.sue_one(c)
        return self.sue_one_output(h)

    def su_t1_decode(self, x):
        h = self.sud_one(x)
        return self.sud_one_out(h)
    
    def su_t2_encode(self, c):
        h = self.sue_two(c)
        return self.sue_two_output(h)

    def su_t2_decode(self, x):
        h = self.sud_two(x)
        return self.sud_two_out(h)
    

class SU_ohne_CUnet(nn.Module):
    def __init__(self, n_x_h, n_x_latent):
        super(SU_ohne_CUnet, self).__init__()

        self.n_x_h = n_x_h
        self.n_x_latent = n_x_latent

        #Encoder
        self.su_ohnecu_enc = nn.Sequential(
            nn.Linear(784, n_x_h), nn.Tanh(),
        )
        self.su_ohnecu_enc_output = nn.Linear(n_x_h, n_x_latent)

        #Decoder
        self.su_ohnecu_dec = nn.Sequential(
            nn.Linear(n_x_latent, 16), nn.Tanh(),
        )
        self.su_ohnecu_dec_out = nn.Linear(16, 1)

    def su_ohnecu_encode(self, s):
        h = self.su_ohnecu_enc(s)
        return self.su_ohnecu_enc_output(h)

    def su_ohnecu_decode(self, x):
        h = self.su_ohnecu_dec(x)
        return self.su_ohnecu_dec_out(h)




class Distributed_MultiTaskMultiUserComm:

    def __init__(self, n_in, n_c_latent, n_c_h, n_x_latent, n_x_h, n_ohne_CU,n_hidden_layer_ohne_CU,n_hidden_layer_su_with_cu,transmit_CU,k_1,k_2,k_3,s_1,s_2,s_3,centralised_observation,SNR_task1,SNR_task2,additional_maxpool,pad,SNR_train_range=0):
        

        # if centralised_observation == True:
        #     n_x_latent = n_x_latent * 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multitask_network = Distributed_MultiTaskNetwork(n_in=n_in, n_c_latent= n_c_latent, n_c_h=n_c_h, n_x_in=n_c_latent, n_x_latent=n_x_latent, n_x_h=n_x_h,n_hidden_layer=n_hidden_layer_su_with_cu,transmit_CU=transmit_CU,k_1=k_1, k_2=k_2, k_3=k_3,centralised_observation=centralised_observation,SNR_task_1=SNR_task1,SNR_task_2=SNR_task2,additional_maxpool=additional_maxpool,pad=pad,SNR_train_range=SNR_train_range).to(self.device)
        self.SUohneCU_network = Distributed_SU_ohne_CUnet(n_x_h=n_ohne_CU, n_x_latent=n_x_latent,n_hidden_layer=n_hidden_layer_ohne_CU,s_1=s_1, s_2=s_2, s_3=s_3,centralised_observation=centralised_observation,SNR=SNR_task1,additional_maxpool=additional_maxpool,pad=pad,SNR_train_range=SNR_train_range).to(self.device)
        self.SUohneCU_network_task2 = Distributed_SU_ohne_CUnet(n_x_h=n_ohne_CU, n_x_latent=n_x_latent,n_hidden_layer=n_hidden_layer_ohne_CU,Nr_decoder_outputs=10,s_1=s_1, s_2=s_2, s_3=s_3,centralised_observation=centralised_observation,SNR=SNR_task2,additional_maxpool=additional_maxpool,pad=pad,SNR_train_range=SNR_train_range).to(self.device)

        self.SUohneCU_network_TEST = Distributed_SU_ohne_CUnet(n_x_h=n_ohne_CU, n_x_latent=n_x_latent,n_hidden_layer=n_hidden_layer_ohne_CU,s_1=s_1, s_2=s_2, s_3=s_3,centralised_observation=centralised_observation,SNR=SNR_task1,additional_maxpool=additional_maxpool,pad=pad,SNR_train_range=SNR_train_range).to(self.device)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.categ_criterion = torch.nn.CrossEntropyLoss() 
        

        self.train_loss_su1 = []
        self.train_loss_su2 = []
        self.train_losses = []
        self.train_times = []
        self.accuracy_task_1 = []
        self.accuracy_task_2 = []
        self.error_task_1 = []
        self.error_task_2 = []

        self.accuracy_task_1_ohnecu = []
        self.error_task_1_ohnecu = []
        self.train_loss_su_ohnecu = []

        self.train_loss_su_ohnecu_task2 = []
        self.accuracy_task_2_ohnecu = []
        self.error_task_2_ohnecu = []
        

        self.error_task_1_mitcu_overSNR = []
        self.error_task_2_mitcu_overSNR = []
        self.error_task_1_ohnecu_overSNR = []
        self.error_task_2_ohnecu_overSNR = []

        self.SNR_train_range = SNR_train_range

    #Distributed case with 4 agents
    def _loss_SU1(self, s_1, s_2, s_3, s_4, targets1):
        c_1 = self.multitask_network.cu_encode_agent_1(s_1)
        x_1 = c_1
        # x_1 = self.multitask_network.su_t1_encode_agent_1(c_1)
        x_hat_1 = awgn_channel(self.multitask_network.SNR_task_1,x_1,SNR_train_range=self.SNR_train_range)
        
        if self.multitask_network.centralised_observation == False:
            c_2 = self.multitask_network.cu_encode_agent_2(s_2)
            x_2 = c_2
            # x_2 = self.multitask_network.su_t1_encode_agent_2(c_2)
            x_hat_2 = awgn_channel(self.multitask_network.SNR_task_1,x_2,SNR_train_range=self.SNR_train_range)

            c_3 = self.multitask_network.cu_encode_agent_3(s_3)
            x_3 = c_3
            # x_3 = self.multitask_network.su_t1_encode_agent_3(c_3)
            x_hat_3 = awgn_channel(self.multitask_network.SNR_task_1,x_3,SNR_train_range=self.SNR_train_range)

            c_4 = self.multitask_network.cu_encode_agent_4(s_4)
            x_4 = c_4
            # x_4 = self.multitask_network.su_t1_encode_agent_3(c_4)
            x_hat_4 = awgn_channel(self.multitask_network.SNR_task_1,x_4,SNR_train_range=self.SNR_train_range)

        if self.multitask_network.centralised_observation == True:
            x_hat = x_hat_1
        else:
            x_hat = torch.cat((x_hat_1,x_hat_2,x_hat_3,x_hat_4),axis=1)
        

        z     = self.multitask_network.su_t1_decode(x_hat)
        targets1 = targets1.float()
        sre = self.criterion(z, targets1.view(-1,1))  # This takes care of sigmoid itself and has a negative
        sre_sum = torch.sum(sre) # semantic recovery error(sre)           
        loss_su1 = sre_sum 
        return loss_su1
    
    # def _loss_SU1_TEST(self, s_1, s_2, s_3, s_4, targets1):
    #     x_1 = self.SUohneCU_network_TEST.su_ohnecu_encode_agent_1(s_1)
    #     x_hat_1 = awgn_channel(x_1)

    #     x_2 = self.SUohneCU_network_TEST.su_ohnecu_encode_agent_2(s_2)
    #     x_hat_2 = awgn_channel(x_2)

    #     x_3 = self.SUohneCU_network_TEST.su_ohnecu_encode_agent_3(s_3)
    #     x_hat_3 = awgn_channel(x_3)

    #     x_4 = self.SUohneCU_network_TEST.su_ohnecu_encode_agent_4(s_4)
    #     x_hat_4 = awgn_channel(x_4)

    #     x_hat = torch.cat((x_hat_1,x_hat_2,x_hat_3,x_hat_4),axis=1)

    #     z = self.SUohneCU_network.su_ohnecu_decode(x_hat)
    #     targets1 = targets1.float()
    #     sre = self.criterion(z, targets1.view(-1,1))  # This takes care of sigmoid itself and has a negative
    #     sre_sum = torch.sum(sre) # semantic recovery error(sre)           
    #     loss_su1 = sre_sum 
    #     return loss_su1


    def _loss_SU2(self, s_1, s_2, s_3, s_4, targets2):
        task = 2

        c_1 = self.multitask_network.cu_encode_agent_1(s_1,task=task)
        x2_1 = c_1
        # x2_1 = self.multitask_network.su_t2_encode_agent_1(c_1)
        x2_hat_1 = awgn_channel(self.multitask_network.SNR_task_2,x2_1,SNR_train_range=self.SNR_train_range)

        if self.multitask_network.centralised_observation == False:
            c_2 = self.multitask_network.cu_encode_agent_2(s_2,task=task)
            x2_2 = c_2
            # x2_2 = self.multitask_network.su_t2_encode_agent_2(c_2)
            x2_hat_2 = awgn_channel(self.multitask_network.SNR_task_2,x2_2,SNR_train_range=self.SNR_train_range)

            c_3 = self.multitask_network.cu_encode_agent_3(s_3,task=task)
            x2_3 = c_3
            # x2_3 = self.multitask_network.su_t2_encode_agent_3(c_3)
            x2_hat_3 = awgn_channel(self.multitask_network.SNR_task_2,x2_3,SNR_train_range=self.SNR_train_range)

            c_4 = self.multitask_network.cu_encode_agent_4(s_4,task=task)
            x2_4 = c_4
            # x2_4 = self.multitask_network.su_t2_encode_agent_4(c_4)
            x2_hat_4 = awgn_channel(self.multitask_network.SNR_task_2,x2_4,SNR_train_range=self.SNR_train_range)

        if self.multitask_network.centralised_observation == True:
            x2_hat = x2_hat_1
        else:
            x2_hat = torch.cat((x2_hat_1,x2_hat_2,x2_hat_3,x2_hat_4),axis=1)

        z2 = self.multitask_network.su_t2_decode(x2_hat)
        sre2 = self.categ_criterion(z2, targets2) 
        sre2_sum = torch.sum(sre2)            
        loss_su2 = sre2_sum 
        return loss_su2
    
    def _loss_SU_ohneCU (self, s_1, s_2, s_3, s_4, targets1):
        x_1 = self.SUohneCU_network.su_ohnecu_encode_agent_1(s_1)
        x_hat_1 = awgn_channel(self.SUohneCU_network.SNR,x_1,SNR_train_range=self.SNR_train_range)

        if self.SUohneCU_network.centralised_observation == False:
            x_2 = self.SUohneCU_network.su_ohnecu_encode_agent_2(s_2)
            x_hat_2 = awgn_channel(self.SUohneCU_network.SNR,x_2,SNR_train_range=self.SNR_train_range)

            x_3 = self.SUohneCU_network.su_ohnecu_encode_agent_3(s_3)
            x_hat_3 = awgn_channel(self.SUohneCU_network.SNR,x_3,SNR_train_range=self.SNR_train_range)

            x_4 = self.SUohneCU_network.su_ohnecu_encode_agent_4(s_4)
            x_hat_4 = awgn_channel(self.SUohneCU_network.SNR,x_4,SNR_train_range=self.SNR_train_range)

        if self.SUohneCU_network.centralised_observation == True:
            x_hat = x_hat_1
        else:
            x_hat = torch.cat((x_hat_1,x_hat_2,x_hat_3,x_hat_4),axis=1)

        z = self.SUohneCU_network.su_ohnecu_decode(x_hat)
        targets1 = targets1.float()
        recovery_error = self.criterion(z, targets1.view(-1,1))
        error_sum = torch.sum(recovery_error)
        loss_su_ohne_cu = error_sum
        return loss_su_ohne_cu
    
    def _loss_SU_ohneCU_task2 (self, s_1, s_2, s_3, s_4, targets2):
        x_1 = self.SUohneCU_network_task2.su_ohnecu_encode_agent_1(s_1)
        x_hat_1 = awgn_channel(self.SUohneCU_network_task2.SNR,x_1,SNR_train_range=self.SNR_train_range)

        if self.SUohneCU_network.centralised_observation == False:
            x_2 = self.SUohneCU_network_task2.su_ohnecu_encode_agent_2(s_2)
            x_hat_2 = awgn_channel(self.SUohneCU_network_task2.SNR,x_2,SNR_train_range=self.SNR_train_range)

            x_3 = self.SUohneCU_network_task2.su_ohnecu_encode_agent_3(s_3)
            x_hat_3 = awgn_channel(self.SUohneCU_network_task2.SNR,x_3,SNR_train_range=self.SNR_train_range)

            x_4 = self.SUohneCU_network_task2.su_ohnecu_encode_agent_4(s_4)
            x_hat_4 = awgn_channel(self.SUohneCU_network_task2.SNR,x_4,SNR_train_range=self.SNR_train_range)

        if self.SUohneCU_network.centralised_observation == True:
            x_hat = x_hat_1
        else:
            x_hat = torch.cat((x_hat_1,x_hat_2,x_hat_3,x_hat_4),axis=1)

        z = self.SUohneCU_network_task2.su_ohnecu_decode(x_hat)

        sre2 = self.categ_criterion(z, targets2) 
        sre2_sum = torch.sum(sre2)            
        loss_su2_ohne_cu = sre2_sum 
        return loss_su2_ohne_cu
    
    


    def fit(self, first_dataset, second_dataset, first_test_dataset, second_test_dataset, batch_size=100,
            n_epoch_primal=500, learning_rate=0.01, path=None, iteration=np.random.randint(0,100),rotate_images = True,low_angle = -20,high_angle = +20):

        
        N = 5000    # It must be 15000 if we repeat the dataset                                            

        first_loader = DataLoader(first_dataset, batch_size=batch_size, shuffle=True)
        second_loader = DataLoader(second_dataset, batch_size=batch_size, shuffle=True)
     

        optimizer = torch.optim.Adam(self.multitask_network.parameters(), lr=learning_rate)
        su_ohne_optimizer = torch.optim.Adam(self.SUohneCU_network.parameters(), lr=learning_rate)
        su_ohne_optimizer_task2 = torch.optim.Adam(self.SUohneCU_network_task2.parameters(), lr=learning_rate)
        su_ohne_optimizer_TEST = torch.optim.Adam(self.SUohneCU_network_TEST.parameters(), lr=learning_rate)

        for epoch_primal in range(n_epoch_primal):
            start = time.time()


            mean_loss_su1 = 0
            mean_loss_su2 = 0
            mean_loss = 0

            mean_loss_su_ohne = 0
            mean_loss_su_ohne_task2 = 0

            zipped_dataloader = zip(first_loader, second_loader)
            self.multitask_network.train()
            self.SUohneCU_network.train()
            self.SUohneCU_network_task2.train()
            self.SUohneCU_network_TEST.train()


            for batch, ((S, targets1), (S2, targets2)) in enumerate(zipped_dataloader):
                
                S, targets1, S2, targets2 = S.squeeze().to(self.device), targets1.to(self.device), S2.squeeze().to(self.device), targets2.to(self.device) 
                
                S_reshaped = S.reshape((batch_size,28, 28))
                S2_reshaped = S2.reshape((batch_size,28, 28))
                
                #split datset for each agent 1/4 of the image
                S_1_reshaped = S_reshaped[:,0:14,0:14]
                S_2_reshaped = S_reshaped[:,0:14,14:]
                S_3_reshaped = S_reshaped[:,14:,0:14]
                S_4_reshaped = S_reshaped[:,14:,14:]

                S2_1_reshaped = S2_reshaped[:,0:14,0:14]
                S2_2_reshaped = S2_reshaped[:,0:14,14:]
                S2_3_reshaped = S2_reshaped[:,14:,0:14]
                S2_4_reshaped = S2_reshaped[:,14:,14:]

                #rotate images
                if rotate_images == True:
                    # for rotation_iter in range(batch_size):
                    #     S_1_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S_1_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S_2_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S_2_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S_3_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S_3_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S_4_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S_4_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))

                    #     S2_1_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S2_1_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S2_2_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S2_2_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S2_3_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S2_3_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S2_4_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S2_4_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    
                    rot_angle_1 = np.random.randint(low_angle, high=high_angle)
                    rot_angle_2 = np.random.randint(low_angle, high=high_angle)
                    rot_angle_3 = np.random.randint(low_angle, high=high_angle)
                    rot_angle_4 = np.random.randint(low_angle, high=high_angle)
                    
                    S_1_reshaped = torchvision.transforms.functional.rotate(S_1_reshaped ,angle=rot_angle_1)
                    S_2_reshaped = torchvision.transforms.functional.rotate(S_2_reshaped ,angle=rot_angle_2)
                    S_3_reshaped = torchvision.transforms.functional.rotate(S_3_reshaped ,angle=rot_angle_3)
                    S_4_reshaped = torchvision.transforms.functional.rotate(S_4_reshaped ,angle=rot_angle_4)

                    S2_1_reshaped = torchvision.transforms.functional.rotate(S2_1_reshaped ,angle=rot_angle_1)
                    S2_2_reshaped = torchvision.transforms.functional.rotate(S2_2_reshaped ,angle=rot_angle_2)
                    S2_3_reshaped = torchvision.transforms.functional.rotate(S2_3_reshaped ,angle=rot_angle_3)
                    S2_4_reshaped = torchvision.transforms.functional.rotate(S2_4_reshaped ,angle=rot_angle_4)

                #flatten dataset 14*14 = 196
                S_1 = S_1_reshaped.reshape((batch_size,196))
                S_2 = S_2_reshaped.reshape((batch_size,196))
                S_3 = S_3_reshaped.reshape((batch_size,196))
                S_4 = S_4_reshaped.reshape((batch_size,196))

                S2_1 = S2_1_reshaped.reshape((batch_size,196))
                S2_2 = S2_2_reshaped.reshape((batch_size,196))
                S2_3 = S2_3_reshaped.reshape((batch_size,196))
                S2_4 = S2_4_reshaped.reshape((batch_size,196))

                if self.multitask_network.centralised_observation == True:
                    S_1 = S
                    S2_1 = S2


                self.multitask_network.zero_grad()
                self.SUohneCU_network_TEST.zero_grad()

                loss_SU1 = self._loss_SU1(S_1,S_2,S_3,S_4, targets1=targets1) 
                mean_loss_su1 += loss_SU1.item() / N

                loss_SU2 = self._loss_SU2(S2_1, S2_2, S2_3, S2_4, targets2=targets2)
                mean_loss_su2 += loss_SU2.item() / N
                # loss_SU2 = 0
                # mean_loss_su2 = 0

                # loss = loss_SU1 * loss_SU2 #=======================================================================================================================================================
                loss = loss_SU1 + loss_SU2
                mean_loss = mean_loss_su1 + mean_loss_su2

                #optimize sum / optimize losses individually
                loss.backward(retain_graph=True)
                # loss_SU2.backward()
                # loss_SU1.backward()

                optimizer.step()
                su_ohne_optimizer_TEST.step()


            for batch, (S, targets1) in enumerate(first_loader):
                S, targets1 = S.squeeze().to(self.device), targets1.to(self.device)
                
                S_reshaped = S.reshape((batch_size,28, 28))
                
                #split datset for each agent 1/4 of the image
                S_1_reshaped = S_reshaped[:,0:14,0:14]
                S_2_reshaped = S_reshaped[:,0:14,14:]
                S_3_reshaped = S_reshaped[:,14:,0:14]
                S_4_reshaped = S_reshaped[:,14:,14:]

                #rotate images
                if rotate_images == True:
                    # for rotation_iter in range(batch_size):
                    #     S_1_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S_1_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S_2_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S_2_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S_3_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S_3_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S_4_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S_4_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    S_1_reshaped = torchvision.transforms.functional.rotate(S_1_reshaped ,angle=rot_angle_1)
                    S_2_reshaped = torchvision.transforms.functional.rotate(S_2_reshaped ,angle=rot_angle_2)
                    S_3_reshaped = torchvision.transforms.functional.rotate(S_3_reshaped ,angle=rot_angle_3)
                    S_4_reshaped = torchvision.transforms.functional.rotate(S_4_reshaped ,angle=rot_angle_4)

                #flatten dataset 14*14 = 196
                S_1 = S_1_reshaped.reshape((batch_size,196))
                S_2 = S_2_reshaped.reshape((batch_size,196))
                S_3 = S_3_reshaped.reshape((batch_size,196))
                S_4 = S_4_reshaped.reshape((batch_size,196))

                if self.SUohneCU_network.centralised_observation == True:
                    S_1 = S


                self.SUohneCU_network.zero_grad()

                loss_SU_ohne = self._loss_SU_ohneCU(S_1,S_2,S_3,S_4, targets1)
                mean_loss_su_ohne += loss_SU_ohne.item() / N
                loss_SU_ohne.backward()
                su_ohne_optimizer.step() 

            for batch, (S2, targets2) in enumerate(second_loader):
                S2, targets2 = S2.squeeze().to(self.device), targets2.to(self.device)

                S2_reshaped = S2.reshape((batch_size,28, 28))
                
                #split datset for each agent 1/4 of the image
                S2_1_reshaped = S2_reshaped[:,0:14,0:14]
                S2_2_reshaped = S2_reshaped[:,0:14,14:]
                S2_3_reshaped = S2_reshaped[:,14:,0:14]
                S2_4_reshaped = S2_reshaped[:,14:,14:]

                #rotate images
                if rotate_images == True:
                    # for rotation_iter in range(batch_size):
                    #     S2_1_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S2_1_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S2_2_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S2_2_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S2_3_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S2_3_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     S2_4_reshaped[rotation_iter,:,:] = torchvision.transforms.functional.rotate(S2_4_reshaped[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))

                    S2_1_reshaped = torchvision.transforms.functional.rotate(S2_1_reshaped ,angle=rot_angle_1)
                    S2_2_reshaped = torchvision.transforms.functional.rotate(S2_2_reshaped ,angle=rot_angle_2)
                    S2_3_reshaped = torchvision.transforms.functional.rotate(S2_3_reshaped ,angle=rot_angle_3)
                    S2_4_reshaped = torchvision.transforms.functional.rotate(S2_4_reshaped ,angle=rot_angle_4)

                #flatten dataset 14*14 = 196
                S2_1 = S2_1_reshaped.reshape((batch_size,196))
                S2_2 = S2_2_reshaped.reshape((batch_size,196))
                S2_3 = S2_3_reshaped.reshape((batch_size,196))
                S2_4 = S2_4_reshaped.reshape((batch_size,196))

                if self.SUohneCU_network_task2.centralised_observation == True:
                    S2_1 = S2

                self.SUohneCU_network_task2.zero_grad()

                loss_SU_ohne_task2 = self._loss_SU_ohneCU_task2(S2_1,S2_2,S2_3,S2_4, targets2)
                mean_loss_su_ohne_task2 += loss_SU_ohne_task2.item() / N
                loss_SU_ohne_task2.backward()
                su_ohne_optimizer_task2.step() 
            
            #self.multitask_network.eval()
            test_accuracy_task_one, test_accuracy_task_two, test_accuracy_task_one_ohne, test_accuracy_task_two_ohne, error_rate_task1, error_rate_task2, error_rate_task1_ohnecu, error_rate_task2_ohnecu = self.eval(first_test_dataset, second_test_dataset,eval_SNR_task_1=self.multitask_network.SNR_task_1,eval_SNR_task_2=self.multitask_network.SNR_task_2,eval_SNR_task_1_noCU=self.SUohneCU_network.SNR,eval_SNR_task_2_noCU=self.SUohneCU_network_task2.SNR,batch_size=batch_size,iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)


            end = time.time()
            self.train_loss_su1.append(mean_loss_su1)
            self.train_loss_su2.append(mean_loss_su2)
            self.train_losses.append(mean_loss)
            self.train_times.append(end - start)
            self.accuracy_task_1.append(test_accuracy_task_one)
            self.accuracy_task_2.append(test_accuracy_task_two)
            self.error_task_1.append(error_rate_task1)
            self.error_task_2.append(error_rate_task2)
            self.train_loss_su_ohnecu.append(mean_loss_su_ohne)
            self.accuracy_task_1_ohnecu.append(test_accuracy_task_one_ohne)
            self.error_task_1_ohnecu.append(error_rate_task1_ohnecu)

            self.train_loss_su_ohnecu_task2.append(mean_loss_su_ohne_task2)
            self.accuracy_task_2_ohnecu.append(test_accuracy_task_two_ohne)
            self.error_task_2_ohnecu.append(error_rate_task2_ohnecu)
            
            

            print(
                f"VAE epoch: {epoch_primal} / Train: {mean_loss:0.3f} / SU1: {mean_loss_su1:0.3f} / SU2: {mean_loss_su2:0.3f} / SU1_ohne: {mean_loss_su_ohne:0.3f}/ SU2_ohne: {mean_loss_su_ohne_task2:0.3f}")
            
            print(
                f"Test Accuracy Task One: {test_accuracy_task_one:.2f}% / Test Accuracy Task Two: {test_accuracy_task_two:.2f}% / Test Accuracy Ohne CU1: {test_accuracy_task_one_ohne:.2f}% / Test Accuracy Ohne CU2: {test_accuracy_task_two_ohne:.2f}%")


        # eval for different SNR after training finished:
        Nr_SNR_evaluations = 41
        SNR_eval_min = -20
        SNR_eval_max = +20
        SNR_eval_linspace = np.linspace(SNR_eval_min,SNR_eval_max,Nr_SNR_evaluations)
        for SNR_eval_iter in range(Nr_SNR_evaluations):
            
            SNR_current = SNR_eval_linspace[SNR_eval_iter]
            test_accuracy_task_one, test_accuracy_task_two, test_accuracy_task_one_ohne, test_accuracy_task_two_ohne, error_rate_task1, error_rate_task2, error_rate_task1_ohnecu, error_rate_task2_ohnecu = self.eval(first_test_dataset, second_test_dataset, eval_SNR_task_1=SNR_current,eval_SNR_task_2=SNR_current, eval_SNR_task_1_noCU=SNR_current, eval_SNR_task_2_noCU=SNR_current, eval_specific_SNR=True,batch_size=batch_size,iteration=iteration,rotate_images=rotate_images,low_angle=low_angle,high_angle=high_angle)

            self.error_task_1_mitcu_overSNR.append(error_rate_task1)
            self.error_task_2_mitcu_overSNR.append(error_rate_task2)
            self.error_task_1_ohnecu_overSNR.append(error_rate_task1_ohnecu)
            self.error_task_2_ohnecu_overSNR.append(error_rate_task2_ohnecu)

    def eval(self, first_test_dataset, second_test_dataset, batch_size= 100, threshold=0.5,eval_SNR_task_1=0,eval_SNR_task_2=0,eval_SNR_task_1_noCU=0,eval_SNR_task_2_noCU=0,eval_specific_SNR=False,iteration=0,rotate_images=True,low_angle=20,high_angle=20):

        iteration = iteration*100

        first_test_loader = DataLoader(first_test_dataset, batch_size=batch_size, shuffle=False)
        second_test_loader = DataLoader(second_test_dataset, batch_size=batch_size, shuffle=False)

        self.multitask_network.eval()
        self.SUohneCU_network.eval()
        
        total_tone = 0
        correct_tone = 0

        total_ttwo = 0
        correct_ttwo = 0

        correct_tone_ohne = 0
        correct_ttwo_ohne = 0

        zipped_testloader = zip(first_test_loader, second_test_loader)

        with torch.no_grad():
            for batch ,((first_test_data, first_test_target),(second_test_data, second_test_target)) in enumerate(zipped_testloader):

                first_test_data = first_test_data.squeeze().to(self.device)
                first_test_target = first_test_target.to(self.device)
                second_test_target = second_test_target.to(self.device)

                #split image into 4 parts
                first_test_data_reshaped = first_test_data.reshape((batch_size,28, 28))

                #split datset for each agent 1/4 of the image
                first_test_data_reshaped_1 = first_test_data_reshaped[:,0:14,0:14]
                first_test_data_reshaped_2 = first_test_data_reshaped[:,0:14,14:]
                first_test_data_reshaped_3 = first_test_data_reshaped[:,14:,0:14]
                first_test_data_reshaped_4 = first_test_data_reshaped[:,14:,14:]

                #rotate images
                if rotate_images == True:
                    # for rotation_iter in range(batch_size):
                    #     first_test_data_reshaped_1[rotation_iter,:,:] = torchvision.transforms.functional.rotate(first_test_data_reshaped_1[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     first_test_data_reshaped_2[rotation_iter,:,:] = torchvision.transforms.functional.rotate(first_test_data_reshaped_2[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     first_test_data_reshaped_3[rotation_iter,:,:] = torchvision.transforms.functional.rotate(first_test_data_reshaped_3[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    #     first_test_data_reshaped_4[rotation_iter,:,:] = torchvision.transforms.functional.rotate(first_test_data_reshaped_4[0,:,:].expand(1,-1,-1),angle=np.random.randint(low_angle, high=high_angle))
                    first_test_data_reshaped_1 = torchvision.transforms.functional.rotate(first_test_data_reshaped_1 ,angle=np.random.randint(low_angle, high=high_angle))
                    first_test_data_reshaped_2 = torchvision.transforms.functional.rotate(first_test_data_reshaped_2 ,angle=np.random.randint(low_angle, high=high_angle))
                    first_test_data_reshaped_3 = torchvision.transforms.functional.rotate(first_test_data_reshaped_3 ,angle=np.random.randint(low_angle, high=high_angle))
                    first_test_data_reshaped_4 = torchvision.transforms.functional.rotate(first_test_data_reshaped_4 ,angle=np.random.randint(low_angle, high=high_angle))



                first_test_data_1 = first_test_data_reshaped_1.reshape((batch_size,196))
                first_test_data_2 = first_test_data_reshaped_2.reshape((batch_size,196))
                first_test_data_3 = first_test_data_reshaped_3.reshape((batch_size,196))
                first_test_data_4 = first_test_data_reshaped_4.reshape((batch_size,196))

                if self.multitask_network.centralised_observation == True:
                    first_test_data_1 = first_test_data


                #common unit at each agent
                c_1 = self.multitask_network.cu_encode_agent_1(first_test_data_1)
                # c_1 = reparameterize(c_mu_1, c_ln_var_1)
                x1_1 = c_1
                # x1_1 = self.multitask_network.su_t1_encode_agent_1(c_1)
                x1_hat_1 = awgn_channel(eval_SNR_task_1,x1_1,eval_specific_SNR=eval_specific_SNR,iteration=iteration)
                x1_hat = x1_hat_1
                if self.multitask_network.centralised_observation == False:
                    c_2 = self.multitask_network.cu_encode_agent_2(first_test_data_2)
                    # c_2 = reparameterize(c_mu_2, c_ln_var_2)
                    c_3 = self.multitask_network.cu_encode_agent_3(first_test_data_3)
                    # c_3 = reparameterize(c_mu_3, c_ln_var_3)
                    c_4 = self.multitask_network.cu_encode_agent_4(first_test_data_4)
                    # c_4 = reparameterize(c_mu_4, c_ln_var_4)
                    # First Task
    
                    x1_2 = c_2
                    x1_3 = c_3
                    x1_4 = c_4
                
                    # x1_2 = self.multitask_network.su_t1_encode_agent_2(c_2)
                    x1_hat_2 = awgn_channel(eval_SNR_task_1,x1_2,eval_specific_SNR=eval_specific_SNR,iteration=iteration+1)
                    # x1_3 = self.multitask_network.su_t1_encode_agent_3(c_3)
                    x1_hat_3 = awgn_channel(eval_SNR_task_1,x1_3,eval_specific_SNR=eval_specific_SNR,iteration=iteration+2)
                    # x1_4 = self.multitask_network.su_t1_encode_agent_4(c_4)
                    x1_hat_4 = awgn_channel(eval_SNR_task_1,x1_4,eval_specific_SNR=eval_specific_SNR,iteration=iteration+3)
                
                    x1_hat = torch.cat((x1_hat_1,x1_hat_2,x1_hat_3,x1_hat_4),axis=1)


                z1 = self.multitask_network.su_t1_decode(x1_hat)

                #DEBUG
                # x1_1 = self.SUohneCU_network_TEST.su_ohnecu_encode_agent_1(first_test_data_1)
                # x1_hat_1 = awgn_channel(x1_1)
                # x1_2 = self.SUohneCU_network_TEST.su_ohnecu_encode_agent_2(first_test_data_2)
                # x1_hat_2 = awgn_channel(x1_2)
                # x1_3 = self.SUohneCU_network_TEST.su_ohnecu_encode_agent_3(first_test_data_3)
                # x1_hat_3 = awgn_channel(x1_3)
                # x1_4 = self.SUohneCU_network_TEST.su_ohnecu_encode_agent_4(first_test_data_4)
                # x1_hat_4 = awgn_channel(x1_4)

                # x1_hat = torch.cat((x1_hat_1,x1_hat_2,x1_hat_3,x1_hat_4),axis=1)
                # z1 = self.SUohneCU_network.su_ohnecu_decode(x1_hat)

                z1_inferred = torch.sigmoid(z1)


                # Second Task
                x2_1 =  self.multitask_network.cu_encode_agent_1(first_test_data_1,task=2)
                # x2_1 = self.multitask_network.su_t2_encode_agent_1(c_1)
                x2_hat_1 = awgn_channel(eval_SNR_task_2,x2_1,eval_specific_SNR=eval_specific_SNR,iteration=iteration+4)
                x2_hat = x2_hat_1

                if self.multitask_network.centralised_observation == False:
                    x2_2 =  self.multitask_network.cu_encode_agent_2(first_test_data_2,task=2)
                    x2_3 =  self.multitask_network.cu_encode_agent_3(first_test_data_3,task=2)
                    x2_4 =  self.multitask_network.cu_encode_agent_4(first_test_data_4,task=2)
                
                    # x2_2 = self.multitask_network.su_t2_encode_agent_2(c_2)
                    x2_hat_2 = awgn_channel(eval_SNR_task_2,x2_2,eval_specific_SNR=eval_specific_SNR,iteration=iteration+5)
                    # x2_3 = self.multitask_network.su_t2_encode_agent_3(c_3)
                    x2_hat_3 = awgn_channel(eval_SNR_task_2,x2_3,eval_specific_SNR=eval_specific_SNR,iteration=iteration+6)
                    # x2_4 = self.multitask_network.su_t2_encode_agent_4(c_4)
                    x2_hat_4 = awgn_channel(eval_SNR_task_2,x2_4,eval_specific_SNR=eval_specific_SNR,iteration=iteration+7)

                    x2_hat = torch.cat((x2_hat_1,x2_hat_2,x2_hat_3,x2_hat_4),axis=1)

                z2 = self.multitask_network.su_t2_decode(x2_hat)
                z2_inferred = torch.softmax(z2, dim=1)


                # First task ohne CU
                x3_1 = self.SUohneCU_network.su_ohnecu_encode_agent_1(first_test_data_1)
                x3_hat_1 = awgn_channel(eval_SNR_task_1_noCU,x3_1,eval_specific_SNR=eval_specific_SNR,iteration=iteration)
                x3_hat = x3_hat_1

                if self.SUohneCU_network.centralised_observation == False:
                    x3_2 = self.SUohneCU_network.su_ohnecu_encode_agent_2(first_test_data_2)
                    x3_hat_2 = awgn_channel(eval_SNR_task_1_noCU,x3_2,eval_specific_SNR=eval_specific_SNR,iteration=iteration+1)
                    x3_3 = self.SUohneCU_network.su_ohnecu_encode_agent_3(first_test_data_3)
                    x3_hat_3 = awgn_channel(eval_SNR_task_1_noCU,x3_3,eval_specific_SNR=eval_specific_SNR,iteration=iteration+2)
                    x3_4 = self.SUohneCU_network.su_ohnecu_encode_agent_4(first_test_data_4)
                    x3_hat_4 = awgn_channel(eval_SNR_task_1_noCU,x3_4,eval_specific_SNR=eval_specific_SNR,iteration=iteration+3)

                    x3_hat = torch.cat((x3_hat_1,x3_hat_2,x3_hat_3,x3_hat_4),axis=1)
                z3 = self.SUohneCU_network.su_ohnecu_decode(x3_hat)
                z3_inferred = torch.sigmoid(z3)

                # Second task ohne CU
                x4_1 = self.SUohneCU_network_task2.su_ohnecu_encode_agent_1(first_test_data_1)
                x4_hat_1 = awgn_channel(eval_SNR_task_2_noCU,x4_1,eval_specific_SNR=eval_specific_SNR,iteration=iteration+4)
                x4_hat = x4_hat_1
                if self.SUohneCU_network_task2.centralised_observation == False:
                    x4_2 = self.SUohneCU_network_task2.su_ohnecu_encode_agent_2(first_test_data_2)
                    x4_hat_2 = awgn_channel(eval_SNR_task_2_noCU,x4_2,eval_specific_SNR=eval_specific_SNR,iteration=iteration+5)
                    x4_3 = self.SUohneCU_network_task2.su_ohnecu_encode_agent_3(first_test_data_3)
                    x4_hat_3 = awgn_channel(eval_SNR_task_2_noCU,x4_3,eval_specific_SNR=eval_specific_SNR,iteration=iteration+6)
                    x4_4 = self.SUohneCU_network_task2.su_ohnecu_encode_agent_4(first_test_data_4)
                    x4_hat_4 = awgn_channel(eval_SNR_task_2_noCU,x4_4,eval_specific_SNR=eval_specific_SNR,iteration=iteration+7)

                    x4_hat = torch.cat((x4_hat_1,x4_hat_2,x4_hat_3,x4_hat_4),axis=1)

                z4 = self.SUohneCU_network_task2.su_ohnecu_decode(x4_hat)
                z4_inferred = torch.softmax(z4,dim=1)


                # Tests
                predicted_t1 = (z1_inferred > threshold).float()
                total_tone += first_test_target.size(0)
                first_test_target = first_test_target.view_as(predicted_t1)  # Ensuring target has the same shape as predicted
                correct_tone += torch.sum((predicted_t1 == first_test_target).float())   

                predicted_t2 = torch.argmax(z2_inferred, dim=1)
                total_ttwo += second_test_target.size(0)
                second_test_target = second_test_target.view_as(predicted_t2)  # Ensuring target has the same shape as predicted
                correct_ttwo += torch.sum((predicted_t2 == second_test_target).float())

                predicted3_t1 = (z3_inferred > threshold).float()
                correct_tone_ohne += torch.sum((predicted3_t1 == first_test_target).float()) 

                predicted4_t2 = torch.argmax(z4_inferred, dim=1)
                correct_ttwo_ohne += torch.sum((predicted4_t2 == second_test_target).float()) 
        
        
        #print('Sum of First predictions:', predicted_t1.sum().item())
        accuracy_task_one = (correct_tone / total_tone) * 100
        error_task_one = 1 - (correct_tone / total_tone)
        #print('Accuracy of the First Task: {:.2f}%'.format(accuracy_task_one))

        #print('Sum of Second predictions:', predicted_t2.sum().item())
        accuracy_task_two = (correct_ttwo / total_ttwo) * 100
        error_task_two = 1 - (correct_ttwo / total_ttwo)
        #print('Accuracy of the Second Task: {:.2f}%'.format(accuracy_task_two))

        accuracy_task_one_ohnecu = (correct_tone_ohne / total_tone) * 100
        error_task_one_ohnecu = 1 - (correct_tone_ohne / total_tone)


        accuracy_task_two_ohnecu = (correct_ttwo_ohne / total_ttwo) * 100
        error_task_two_ohnecu = 1 - (correct_ttwo_ohne / total_ttwo)

        

        gc.collect()
        torch.cuda.empty_cache()
        
        return accuracy_task_one, accuracy_task_two, accuracy_task_one_ohnecu, accuracy_task_two_ohnecu, error_task_one, error_task_two, error_task_one_ohnecu, error_task_two_ohnecu


class MultiTaskMultiUserComm:

    def __init__(self, n_in, n_c_latent, n_c_h, n_x_latent, n_x_h):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multitask_network = MultiTaskNetwork(n_in=n_in, n_c_latent= n_c_latent, n_c_h=n_c_h, n_x_in=n_c_latent, n_x_latent=n_x_latent, n_x_h=n_x_h).to(self.device)
        self.SUohneCU_network = SU_ohne_CUnet(n_x_h=n_x_h, n_x_latent=n_x_latent).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.categ_criterion = torch.nn.CrossEntropyLoss() 
        

        self.train_loss_su1 = []
        self.train_loss_su2 = []
        self.train_losses = []
        self.train_times = []
        self.accuracy_task_1 = []
        self.accuracy_task_2 = []
        self.error_task_1 = []
        self.error_task_2 = []

        self.accuracy_task_1_ohnecu = []
        self.error_task_1_ohnecu = []
        self.train_loss_su_ohnecu = []
              


    def _loss_SU1(self, s, targets1):
        c_mu, c_ln_var = self.multitask_network.cu_encode(s)
        c = reparameterize(c_mu, c_ln_var)
        x = self.multitask_network.su_t1_encode(c)
        x_hat = awgn_channel(self.multitask_network.SNR,x)
        z     = self.multitask_network.su_t1_decode(x_hat)
        targets1 = targets1.float()
        sre = self.criterion(z, targets1.view(-1,1))  # This takes care of sigmoid itself and has a negative
        sre_sum = torch.sum(sre) # semantic recovery error(sre)           
        loss_su1 = sre_sum 
        return loss_su1


    def _loss_SU2(self, s, targets2):
        c_mu, c_ln_var = self.multitask_network.cu_encode(s)
        c = reparameterize(c_mu, c_ln_var)
        x2 = self.multitask_network.su_t2_encode(c)
        x2_hat = awgn_channel(self.multitask_network.SNR,x2)
        z2 = self.multitask_network.su_t2_decode(x2_hat)
        sre2 = self.categ_criterion(z2, targets2) 
        sre2_sum = torch.sum(sre2)            
        loss_su2 = sre2_sum 
        return loss_su2
    
    def _loss_SU_ohneCU (self, s, targets1):
        x = self.SUohneCU_network.su_ohnecu_encode(s)
        x_hat = awgn_channel(self.multitask_network.SNR,x)
        z = self.SUohneCU_network.su_ohnecu_decode(x_hat)
        targets1 = targets1.float()
        recovery_error = self.criterion(z, targets1.view(-1,1))
        error_sum = torch.sum(recovery_error)
        loss_su_ohne_cu = error_sum
        return loss_su_ohne_cu

    

    def fit(self, first_dataset, second_dataset, first_test_dataset, second_test_dataset, batch_size=100,
            n_epoch_primal=500, learning_rate=0.01, path=None):

        
        N = 5000    # It must be 15000 if we repeat the dataset                                            

        first_loader = DataLoader(first_dataset, batch_size=batch_size, shuffle=True)
        second_loader = DataLoader(second_dataset, batch_size=batch_size, shuffle=True)
     

        optimizer = torch.optim.Adam(self.multitask_network.parameters(), lr=learning_rate)
        su_ohne_optimizer = torch.optim.Adam(self.SUohneCU_network.parameters(), lr=learning_rate)

        for epoch_primal in range(n_epoch_primal):
            start = time.time()


            mean_loss_su1 = 0
            mean_loss_su2 = 0
            mean_loss = 0

            mean_loss_su_ohne = 0

            zipped_dataloader = zip(first_loader, second_loader)
            self.multitask_network.train()
            self.SUohneCU_network.train()


            for batch, ((S, targets1), (S2, targets2)) in enumerate(zipped_dataloader):
                
                S, targets1, S2, targets2 = S.squeeze().to(self.device), targets1.to(self.device), S2.squeeze().to(self.device), targets2.to(self.device) 
                

                self.multitask_network.zero_grad()

                loss_SU1 = self._loss_SU1(S, targets1=targets1) 
                mean_loss_su1 += loss_SU1.item() / N

                loss_SU2 = self._loss_SU2(S2, targets2=targets2)
                mean_loss_su2 += loss_SU2.item() / N

                loss = loss_SU1 + loss_SU2
                mean_loss = mean_loss_su1 + mean_loss_su2

                loss.backward(retain_graph=True)
                optimizer.step()


            for batch, (S, targets1) in enumerate(first_loader):
                S, targets1 = S.squeeze().to(self.device), targets1.to(self.device)

                self.SUohneCU_network.zero_grad()

                loss_SU_ohne = self._loss_SU_ohneCU(S, targets1)
                mean_loss_su_ohne += loss_SU_ohne.item() / N
                loss_SU_ohne.backward()
                su_ohne_optimizer.step() 

            
            #self.multitask_network.eval()
            test_accuracy_task_one, test_accuracy_task_two, test_accuracy_task_one_ohne, error_rate_task1, error_rate_task2, error_rate_task1_ohnecu = self.eval(first_test_dataset, second_test_dataset)


            end = time.time()
            self.train_loss_su1.append(mean_loss_su1)
            self.train_loss_su2.append(mean_loss_su2)
            self.train_losses.append(mean_loss)
            self.train_times.append(end - start)
            self.accuracy_task_1.append(test_accuracy_task_one)
            self.accuracy_task_2.append(test_accuracy_task_two)
            self.error_task_1.append(error_rate_task1)
            self.error_task_2.append(error_rate_task2)
            self.train_loss_su_ohnecu.append(mean_loss_su_ohne)
            self.accuracy_task_1_ohnecu.append(test_accuracy_task_one_ohne)
            self.error_task_1_ohnecu.append(error_rate_task1_ohnecu)


            print(
                f"VAE epoch: {epoch_primal} / Train: {mean_loss:0.3f} / SU1: {mean_loss_su1:0.3f} / SU2: {mean_loss_su2:0.3f} / SU_ohne: {mean_loss_su_ohne:0.3f}")
            
            print(
                f"Test Accuracy Task One: {test_accuracy_task_one:.2f}% / Test Accuracy Task Two: {test_accuracy_task_two:.2f}% / Test Accuracy Ohne CU: {test_accuracy_task_one_ohne:.2f}%")







    def eval(self, first_test_dataset, second_test_dataset, batch_size= 100, threshold=0.5):

        first_test_loader = DataLoader(first_test_dataset, batch_size=batch_size, shuffle=False)
        second_test_loader = DataLoader(second_test_dataset, batch_size=batch_size, shuffle=False)

        self.multitask_network.eval()
        self.SUohneCU_network.eval()
        
        total_tone = 0
        correct_tone = 0

        total_ttwo = 0
        correct_ttwo = 0

        correct_tone_ohne = 0

        zipped_testloader = zip(first_test_loader, second_test_loader)

        with torch.no_grad():
            for batch ,((first_test_data, first_test_target),(second_test_data, second_test_target)) in enumerate(zipped_testloader):

                first_test_data = first_test_data.squeeze().to(self.device)
                first_test_target = first_test_target.to(self.device)
                second_test_target = second_test_target.to(self.device)

                c_mu, c_ln_var = self.multitask_network.cu_encode(first_test_data)
                c = reparameterize(c_mu, c_ln_var)
                # First Task
                x1 = self.multitask_network.su_t1_encode(c)
                x1_hat = awgn_channel(self.multitask_network.SNR,x1)
                z1 = self.multitask_network.su_t1_decode(x1_hat)
                z1_inferred = torch.sigmoid(z1)
                # Second Task
                x2 = self.multitask_network.su_t2_encode(c)
                x2_hat = awgn_channel(self.multitask_network.SNR,x2)
                z2 = self.multitask_network.su_t2_decode(x2_hat)
                z2_inferred = torch.softmax(z2, dim=1)
                # First aber ohne CU
                x3 = self.SUohneCU_network.su_ohnecu_encode(first_test_data)
                x3_hat = awgn_channel(self.multitask_network.SNR,x3)
                z3 = self.SUohneCU_network.su_ohnecu_decode(x3_hat)
                z3_inferred = torch.sigmoid(z3)

                # Tests
                predicted_t1 = (z1_inferred > threshold).float()
                total_tone += first_test_target.size(0)
                first_test_target = first_test_target.view_as(predicted_t1)  # Ensuring target has the same shape as predicted
                correct_tone += torch.sum((predicted_t1 == first_test_target).float())   

                predicted_t2 = torch.argmax(z2_inferred, dim=1)
                total_ttwo += second_test_target.size(0)
                second_test_target = second_test_target.view_as(predicted_t2)  # Ensuring target has the same shape as predicted
                correct_ttwo += torch.sum((predicted_t2 == second_test_target).float())

                predicted3_t1 = (z3_inferred > threshold).float()
                correct_tone_ohne += torch.sum((predicted3_t1 == first_test_target).float()) 
        
        
        #print('Sum of First predictions:', predicted_t1.sum().item())
        accuracy_task_one = (correct_tone / total_tone) * 100
        error_task_one = 1 - (correct_tone / total_tone)
        #print('Accuracy of the First Task: {:.2f}%'.format(accuracy_task_one))

        #print('Sum of Second predictions:', predicted_t2.sum().item())
        accuracy_task_two = (correct_ttwo / total_ttwo) * 100
        error_task_two = 1 - (correct_ttwo / total_ttwo)
        #print('Accuracy of the Second Task: {:.2f}%'.format(accuracy_task_two))

        accuracy_task_one_ohnecu = (correct_tone_ohne / total_tone) * 100
        error_task_one_ohnecu = 1 - (correct_tone_ohne / total_tone)

        return accuracy_task_one, accuracy_task_two, accuracy_task_one_ohnecu, error_task_one, error_task_two, error_task_one_ohnecu
    
