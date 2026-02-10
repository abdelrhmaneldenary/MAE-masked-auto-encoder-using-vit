import torch
from torch import nn
from patch_embed import Patch_embedding
from masking import Masking
from vit import TransformerBlock

class Encoder(nn.Module):
    def __init__(self, depth=12, embed_dim=768, heads=12):
        super().__init__()
        self.embedding = Patch_embedding()
        self.masking = Masking(masking_ratio=0.75)
        self.blocks = nn.ModuleList([
            TransformerBlock(heads=heads, dim=embed_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, ids_restore = self.masking(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, ids_restore

class Decoder(nn.Module):
    def __init__(self, depth=8, embed_dim=768, decoder_dim=512, heads=8, num_patches=196):
        super().__init__()
        self.embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim)) 
        self.blocks = nn.ModuleList([
            TransformerBlock(heads=heads, dim=decoder_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, 16**2 * 3)

    def forward(self, x, ids_restore):
        x = self.embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = x_ + self.decoder_pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.pred(x)
        return x

class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def patchify(self, imgs):
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, x):
        latent, ids_restore = self.encoder(x)
        pred = self.decoder(latent, ids_restore)
        target = self.patchify(x)
        
        mask = torch.ones([x.shape[0], 196], device=x.device)
        len_keep = latent.shape[1]
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask

if __name__ == "__main__":
    model = MAE()
    dummy_img = torch.randn(2, 3, 224, 224)
    loss, pred, mask = model(dummy_img)
    print(loss.item())
    print(pred.shape)
    print(mask.shape)