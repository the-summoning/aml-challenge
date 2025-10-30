from typing import Optional
from torch import nn
from torch.nn import functional as F
import torch

### OLD MODEL ###

# class Translator(nn.Module):
#     def __init__(self, pad: bool, dim_imgs: int = 1536, dim_text: int = 1024,  mode: str ='linear'):
#         super().__init__()
#         assert mode in ['linear', 'affine', 'isometry'], f'Mode "{mode}" not supported'

#         self.mode = mode
#         use_bias = mode == 'affine'
#         if pad:
#             dim = max(dim_imgs, dim_text)
#             self.linear = nn.Linear(dim, dim, bias=use_bias)

#         else:
#             self.linear = nn.Linear(dim_text, dim_imgs, bias=use_bias)

#     def forward(self, x):
#         return self.linear(x)

#     @torch.no_grad()
#     def orthogonalize(self):
#         assert self.mode == 'isometry', 'Cannot be called for modes != isometry'

#         W = self.linear.weight.data
#         U, _, Vh = torch.linalg.svd(W, full_matrices=False)
#         self.linear.weight.data.copy_(U @ Vh)

class Translator(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1536, mode='affine', use_relative=False, anchors: Optional[torch.Tensor] = None):
        super().__init__()
        assert mode in ['linear', 'affine', 'isometry'], f'Mode "{mode}" not supported'
        assert input_dim > 0 and output_dim > 0, "Expecting positive dimensions"
        assert not use_relative or isinstance(anchors, torch.Tensor) , 'Anchors must be set if using relative representations'
        assert anchors is None or (anchors.ndim == 2 and anchors.shape[0] > 0), '2D Anchors must be provided if using relative representations'
        
        self.mode = mode
        self.use_relative = use_relative
        self.anchors = anchors
        
        self.linear = nn.Linear(
            anchors.shape[0] if self.use_relative else input_dim,
            output_dim,
            bias=self.mode == 'affine'
        )
    
    def compute_relative(self, x):
        assert self.anchors is not None, 'Anchors must be set by calling "set_anchors"'
        
        return F.normalize(x, p=2, dim=1) @ F.normalize(self.anchors.T)
        
    def forward(self, x):
        if self.use_relative:
            x = self.compute_relative(x)
        
        return self.linear(x)
