import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self,dim=768):
        super().__init__()
        self.mlp_operation=nn.Sequential(
        nn.Linear(dim,4*dim),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(4*dim,dim),
        nn.Dropout()
        )
        
    def forward(self,x):
        x=self.mlp_operation(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self,heads=8,dim=768):
        super().__init__()
        self.norm_1=nn.LayerNorm(dim)
        self.norm_2=nn.LayerNorm(dim)
        self.attention=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.mlp=MLP(dim)


    def forward(self,x):
        norm1_x=self.norm_1(x)
        attended_x=self.attention(norm1_x,norm1_x,norm1_x)
        x_resid_1=x+attended_x[0]
        norm2_x=self.norm_2(x_resid_1)
        mlp_x=self.mlp(norm2_x)
        x_resid_2=x_resid_1+mlp_x
        return x_resid_2
    

if __name__=="__main__":
    dummy=torch.randn(20,196,768)
    transfotm=TransformerBlock()
    result=transfotm(dummy)
    print(dummy.shape)
    print(result.shape)
