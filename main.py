from parameter import *
from trainer import Trainer
from tester import Tester

from torch.backends import cudnn
from utils import make_folder

def main(config):
    cudnn.benchmark = True

    if(config.mode=='train'):   
        make_folder(config.root_path,config.version,config.model_save_path)
        make_folder(config.root_path,config.version,config.log_path)

        trainer = Trainer(config)
        trainer.train()

    elif(config.mode=='test'):    
        tester = Tester(config)
        tester.test()
    else:
        print('=======the parameter mode is needed=================')

if __name__ == '__main__':
    config = get_parameters()
    config.mode ='train'
    if(config.mode=='train'):
        config.dataset = 'train_data'
    if(config.mode=='test'):
        config.dataset = 'test_data'
    main(config)