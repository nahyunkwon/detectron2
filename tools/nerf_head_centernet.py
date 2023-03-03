import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict



class NeRF_Head_RetinaNet(nn.Module):
    def __init__(self, nerfWeights, flagConv4feats=False, actMapDim=256):
        # NeRFWeights: 
        #    C x d (C classes x feature dimension), 
        #    format: torch tensor in cpu
        # featTransform: whether to apply more conv layers to the input activation maps
        super(NeRF_Head_RetinaNet, self).__init__()
        
        self.C_classes = nerfWeights.shape[0]  # C classes
        self.nerfFeaDim = nerfWeights.shape[1] # the dimension of vectorized NeRF weights, e.g., 10000
        self.head_param = nerfWeights[:, :, None, None]  # C x 10000 x 1 x 1

        self.flagConv4feats = flagConv4feats
        self.actMapDim = actMapDim  # e.g., 1024 based on FasterRCNN activation maps
        
        self.convLayer4nerfWeightsTransform = nn.Conv2d(self.nerfFeaDim, self.actMapDim, kernel_size=1)
        
        # for the background class, appended to the output feature maps and yielding (C+1) feature maps
        self.backgroundClassifier = nn.Conv2d(self.actMapDim, 1, kernel_size=1)  
        
        # whether to apply more conv layers to the input activation maps; need to be nonlinear otherwise not needed.
        if self.flagConv4feats:
            print('add more conv layer')
            self.convLayer4feaTransform = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(self.actMapDim, self.actMapDim, 1)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(self.actMapDim, self.actMapDim, 1))
            ]))

    def forward(self, featMap):
        # self.head_param: C x 10000 x 1 x 1  -->  C x 1024 x 1 x 1
        self.classifier_weights = self.convLayer4nerfWeightsTransform(self.head_param)
        self.classifier_weights = self.classifier_weights.squeeze().squeeze()  # C x 1024
        self.backgroundFeatMap = self.backgroundClassifier(featMap)
        
        if self.flagConv4feats:
            # featMap: N x 1024 x H x W
            self.featMap = self.convLayer4feaTransform(featMap)
        else:
            self.featMap = featMap 
            
        NN, CC, HH, WW = self.featMap.shape # N x 1024 x H x W
        self.featMap = torch.reshape(self.featMap, (NN, CC, HH*WW)) # N x 1024 x HW
        self.featMap = torch.permute(self.featMap, (0,2,1)) # N x HW x 1024
        
        self.classifier_weights = torch.permute(self.classifier_weights, (1, 0)) # 1024 x C
        
        out = torch.matmul(self.featMap, self.classifier_weights) # N x HW x C
        out = torch.permute(out, [0, 2, 1]) # N x C x HW
        out = torch.reshape(out, (NN, -1, HH, WW)) # N x C x H x W
        out = torch.cat((out, self.backgroundFeatMap), 1)  # concatenate background logit score map, so N x (C+1) x H x W
        return out

    
# simulate some examples    
nerfWeights = torch.randn(100, 256) 
featMap = torch.randn(7, 256, 64, 64) # suppose there are 7 examples, 


# testing the code
nerfHead = NeRF_Head_RetinaNet(nerfWeights, flagConv4feats=False)

logitScoreMap = nerfHead(featMap)
# print(logitScoreMap)
print(logitScoreMap.shape)
print(nerfHead)
print(type(nerfHead))
