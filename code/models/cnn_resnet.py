# importing some dependencies
import torch
import torch.nn as nn
import pretrainedmodels


# In this file we are going to get create a function to get a pretrained model
# from pytorch, so that we can apply transfer learning on top of it.

def get_pretrained(model_name, pretrained,dataset='imagenet'):
    if pretrained:
        model=pretrainedmodels.__dict__[model_name](pretrained=dataset)
    else:
        model=pretrainedmodels.__dict__[model_name](pretrained=None)
    
    model.last_linear=nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.Linear(in_features=512, out_features=2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1), 
        nn.Dropout(p=0.5), 
        nn.Linear(in_features=2048, out_features=37)
    )

    return model




