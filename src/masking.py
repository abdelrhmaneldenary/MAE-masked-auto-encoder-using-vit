import torch
from torch import nn
class Masking(nn.Module):
    def __init__(self, masking_ratio=0.75):
        super().__init__()
        self.masking_ratio = masking_ratio  # Fixed typo: ration -> ratio
    
    def forward(self, x):
        batch_number, seq_len, dim = x.shape
        random_numbers = torch.randn(batch_number, seq_len, device=x.device)
        indices = torch.argsort(random_numbers, dim=1)
        indices_restore = torch.argsort(indices, dim=1)
        # 4. Expand for Gathering (Make it 3D to match x)
        indices_expand = indices.unsqueeze(-1).expand(-1, -1, dim)
        x_shuffled = torch.gather(input=x, dim=1, index=indices_expand)
        len_keep = int((1 - self.masking_ratio) * seq_len)
        x_masked = x_shuffled[:, :len_keep, :]
        
        return x_masked, indices_restore
    
if __name__=="__main__":
    dummy=torch.randn(20,196,768)
    masking_layer=Masking()
    x_masked,indices_restor=masking_layer.forward(dummy)
    print(x_masked.shape)
    print(indices_restor.shape)

