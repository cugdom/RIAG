import time
import torch
import datetime
import torch.nn as nn
from data_loader import Data_Loader
from utils import *
from model_G import *  # Generator
from model_D import *  # Discriminator

class Tester(object):
    def __init__(self,config):
        self.model = config.model
        self.imsize = config.imsize
        self.batch_size = config.batch_size
        self.load_model = config.load_model
        self.shuffle = config.shuffle
        self.dataset = config.dataset
        self.root_path = config.root_path
        self.adv_path = config.test_image_path
        self.log_path = config.log_path
        self.model_save_path = config.model_save_path

        # Path
        self.log_path = os.path.join(self.root_path,self.log_path)
        self.model_save_path = os.path.join(self.root_path,self.model_save_path)

        self.build_model()

        self.start_epoch = 0
        self.start_batch = 0
        if self.load_model:       
            self.load_pretrained_model() 

    def test(self):

        start_time = time.time()

        adv_data_loader = Data_Loader(self.dataset, self.adv_path, self.imsize, self.batch_size, shuf=self.shuffle)
        adv_train_data = adv_data_loader.loader()
       
        imgs_data = adv_train_data.dataset
        for i in range(len(imgs_data)):
            self.G.eval()  
            img_data = imgs_data[i][0]     
            img_folders = imgs_data.imgs[i][0].split('\\')
            img_name = img_folders[-1]
            make_folder(os.path.join(self.sample_root_path,img_folders[-4]),img_folders[-3],img_folders[-2])    
            sample_save_path = os.path.join(self.sample_root_path,img_folders[-4],img_folders[-3],img_folders[-2])
            img_data = img_data.unsqueeze(dim=0)   
            gen_image = self.G(img_data)     
            prediction= self.D.get_class(gen_image)        
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            str_log = "Time [{}] , Elapsed [{}], tar_model [{}], tar_algorithm [{}],image_name [{}] ".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),elapsed, img_folders[-4],img_folders[-3], img_folders[-1])
            print(str_log)
            with open(os.path.join(self.log_path,'log_test.txt'), "a", encoding='utf-8')as f:
                f.write(str_log+'\n')
            
    def build_model(self):
        self.D = Disc('vgg19')       
        if(torch.cuda.is_available()):
            self.G = Gener().cuda()     
        else:
            self.G = Gener()    

    def load_pretrained_model(self):
        checkpoint_list = os.listdir(self.model_save_path)
        if(len(checkpoint_list)>0):   
            checkpoint_list.sort(key=lambda x:os.path.getmtime(os.path.join(self.model_save_path,x)))    
            latest_file_name = checkpoint_list[-1]           
            checkpoint_path = os.path.join(self.model_save_path, latest_file_name)
            load_ckpt = torch.load(checkpoint_path) 
            self.G.load_state_dict(load_ckpt['parameter'])
            print('loaded trained models: (the epoch: {}, the batch: {})..!'.format(load_ckpt['epoch'],load_ckpt['batch']))
