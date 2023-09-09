import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class Disc():

    def __init__(self,model_name):      
        super(Disc,self).__init__()
        if(model_name =='vgg19'):
            if(torch.cuda.is_available()): 
                self.model_pre = models.vgg19(pretrained=True).cuda()    
            else:
                self.model_pre = models.vgg19(pretrained=True)

    def get_feature(self,x):  
        m = create_feature_extractor(self.model_pre,['features.34'])    
        m.eval()      
        out_feature_34 = m(x)
        out = out_feature_34['features.34']
        return out
    
    def get_class(self,x):
        m = self.model_pre.eval()
        out = m(x)

        with open(r'..\imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
        
        _, indices = torch.sort(out, descending=True)      
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100     
        prediction = [classes[indices[0][0]],percentage[indices[0][0]].item()]       
        return prediction
