import os
import time
import torch
import datetime
import torch.nn as nn
from data_loader import Data_Loader
from utils import *
from model_G import *  # Generator
from model_D import *       # Discriminator

class Trainer(object):
    def __init__(self,config):
        self.model = config.model
        self.imsize = config.imsize
        self.batch_size = config.batch_size
        self.total_epochs = config.total_epochs
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.load_model = config.load_model
        self.shuffle = config.shuffle
        self.dataset = config.dataset
        self.root_path = config.root_path
        self.org_path = config.org_image_path
        self.adv_path = config.adv_image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path
        self.log_path = os.path.join(self.root_path,self.log_path)
        self.model_save_path = os.path.join(self.root_path,self.model_save_path)
        
        self.build_model()

        self.start_epoch = 0
        self.start_batch = 0
        if self.load_model:       
            self.load_pretrained_model() 

    def train(self):
        start_time = time.time()
        for epoch in range(self.start_epoch, self.total_epochs):
                       
            adv_data_loader = Data_Loader(self.dataset, self.adv_path, self.imsize, self.batch_size, shuf=self.shuffle)
            adv_train_data = adv_data_loader.loader()

            org_data_loader = Data_Loader(self.dataset, self.org_path, self.imsize, self.batch_size, shuf=self.shuffle)
            org_train_data = org_data_loader.loader(len(adv_train_data.dataset))   

            org_data_iter = iter(org_train_data)
            adv_data_iter = iter(adv_train_data)
            
            batch_total = len(adv_train_data)
            print('The batch of the train is {}'.format(batch_total))
            for batch in range(self.start_batch,batch_total):
                self.G.train()  

                org_images, org_lbls = next(org_data_iter)   
                adv_images, adv_lbls = next(adv_data_iter)
                org = tensor2var(org_images)   
                extracted_feature_org= self.D.get_feature(org)     
                adv = tensor2var(adv_images)                       
                gen_images = self.G(adv)     
                extracted_feature_gen= self.D.get_feature(gen_images)        

                diff = extracted_feature_gen - extracted_feature_org     
                gen_loss = 0.0061 * torch.mean(torch.sum(torch.square(diff), axis=[3]))    
                self.reset_grad()      
                gen_loss.backward()       
                self.g_optimizer.step()   

                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                str_log = "Time [{}] , Elapsed [{}], batch [{}/{}], epoch [{}/{}],gen_loss [{:.4f}] ".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),elapsed, batch+1,batch_total,epoch+1, self.total_epochs,gen_loss)
                print(str_log)
                with open(os.path.join(self.log_path,'log_train.txt'), "a", encoding='utf-8')as f:
                    f.write(str_log+'\n')

                if(batch%2==0):       
                    checkpoint_G = {'parameter': self.G.state_dict(),
                                    'optimizer': self.g_optimizer.state_dict(),                                 
                                    'batch':batch,
                                    'epoch': epoch}
                    torch.save(checkpoint_G, os.path.join(self.model_save_path, 'epoch_{}_batch_{}_G.pth'.format(epoch,batch)))
                if(epoch==self.total_epochs-1 and batch==batch_total-1):    
                    checkpoint_G = {'parameter': self.G.state_dict(),
                                    'optimizer': self.g_optimizer.state_dict(),
                                    'batch':batch,
                                    'epoch': epoch}
                    torch.save(checkpoint_G, os.path.join(self.model_save_path, 'epoch_{}_batch_{}_G.pth'.format(epoch,batch)))                    

    def build_model(self):

        self.D = Disc('vgg19')      
        if(torch.cuda.is_available()):
            self.G = Gener().cuda()      
        else:
            self.G = Gener()             
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])

    def load_pretrained_model(self):
        checkpoint_list = os.listdir(self.model_save_path)
        if(len(checkpoint_list)>0):   
            checkpoint_list.sort(key=lambda x:os.path.getmtime(os.path.join(self.model_save_path,x)))     
            latest_file_name = checkpoint_list[-1]    
            
            checkpoint_path = os.path.join(self.model_save_path, latest_file_name)
            load_ckpt = torch.load(checkpoint_path) 
            self.G.load_state_dict(load_ckpt['parameter'])
            self.g_optimizer.load_state_dict(load_ckpt['optimizer'])
            self.start_epoch = load_ckpt['epoch']
            self.start_batch = load_ckpt['batch']+1
            print('loaded trained models: (the epoch: {}, the batch: {})..!'.format(load_ckpt['epoch'],load_ckpt['batch']))

    def reset_grad(self):   
        self.g_optimizer.zero_grad()
