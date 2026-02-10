import os
import torch
from torch import nn
class Patch_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(3,768,kernel_size=16,stride=16) #used conv here to break the image into patched ,768 dimension of vit input,16 patch number
        self.flatten=nn.Flatten(2,3) #make the dimension as vit input
        self.position_embeddinng=nn.Parameter(torch.randn(1,196,768)) #learnable positional embedding
    def forward(self,x):
        x=self.conv(x)
        x=self.flatten(x)
        x=torch.permute(x,(0,2,1))
        y=self.position_embeddinng
        x=x+y
        return x

if __name__=="__main__":
    dummy_img=torch.randn(20,3,224,224)
    model=Patch_embedding()
    output=model(dummy_img)
    print(output.shape)
    



