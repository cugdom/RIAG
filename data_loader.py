import torch
import torchvision.datasets as dsets
from torchvision import transforms

class Data_Loader():
    def __init__(self,dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf

    def transform(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset

    def load_train_data(self,master_dataset_len):     
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        while(master_dataset_len>len(dataset)):
            dataset = dataset+dataset      
        return dataset

    def load_test_data(self):      
        transforms = self.transform(True, True, True, True)
        dataset = dsets.ImageFolder(self.path, transform=transforms)
        return dataset    
    
    def loader(self,master_dataset_len=0):
        if self.dataset == 'train_data':
            dataset = self.load_train_data(master_dataset_len)
        elif self.dataset == 'test_data':
            dataset = self.load_test_data()                   
        elif self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        print('The lenth of dataset is {}'.format(len(dataset)))
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuf,
                                              num_workers=2,
                                              drop_last=False)
        return loader     

