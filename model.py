from torch import nn
import torch

class Translator(nn.Module):
    def __init__(self, pad: bool, dim_imgs: int = 1536, dim_text: int = 1024,  mode: str ='linear'):
        super().__init__()
        assert mode in ['linear', 'affine', 'isometry'], f'Mode "{mode}" not supported'

        self.mode = mode
        use_bias = mode == 'affine'
        if pad:
            dim = max(dim_imgs, dim_text)
            self.linear = nn.Linear(dim, dim, bias=use_bias)

        else:
            self.linear = nn.Linear(dim_text, dim_imgs, bias=use_bias)

    def forward(self, x):
        return self.linear(x)

    @torch.no_grad()
    def orthogonalize(self):
        assert self.mode == 'isometry', 'Cannot be called for modes != isometry'

        W = self.linear.weight.data
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        self.linear.weight.data.copy_(U @ Vh)
