"""
Projected Transformer - A generalized transformer model with geometric projections.
This model includes projection hooks that can be customized for different geometries.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Callable

class ExponentialTransformer(nn.Module):
    """
    Projected Transformer model that works with any input dimension.
    
    This transformer includes projection hooks that can be customized for different
    geometries (sphere, SO(3), SE(3), etc.). By default, projections are identity
    functions (no-op) and can be overridden or passed as callable functions.
    
    Parameters
    ----------
    input_size : int
        Dimension of input features. Can be any positive integer.
        Examples:
            - Sphere: 3 (3D points)
            - SO(3): 9 (flattened 3x3 rotation matrices)
            - KITTI: 16 (flattened 4x4 SE(3) matrices) or any other dimension
            - Protein: variable (depends on representation)
    nhead : int, default=8
        Number of attention heads.
    d_hid : int, default=2048
        Dimension of the feedforward network.
    nlayers : int, default=6
        Number of transformer encoder layers.
    dropout : float, default=0.5
        Dropout probability.
    dt : float, default=1.0
        Initial value for learnable residual connection scaling factor.
    m : int, default = 3
        The is the dimension of the manifold for a lie group and for the sphere
        it is one plus the dimension
    use_internal_exp : bool, default=True
        If True, applies exp between transformer layers.
        If False, skips internal exp (only end exp is applied).
    exp_func : Callable or None, default=None
        Custom exp function to use between layers.
    """
    def __init__(
        self, 
        input_dim: int, 
        nhead: int = 8, 
        d_hid: int = 2048,
        nlayers: int = 6, 
        dropout: float = 0.5, 
        dt: float = 1.0,
        m: int = 3, 
        use_internal_exp: bool = True,
        internal_exp_func: Optional[Callable] = None,
        end_exp_func: Optional[Callable] = None
    ):
        super().__init__()
        self.model_type = 'ExponentialTransformer'
        self.input_dim = input_dim
        self.m = m
        self.nlayers = nlayers
        self.use_internal_exp = use_internal_exp
        
        # Store custom exp functions
        self.internal_exp_func = internal_exp_func
        self.end_exp_func = end_exp_func
       
        # Transformer encoder layers
        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(input_dim, nhead, d_hid, dropout, batch_first=False)
            for _ in range(nlayers)
        ])
       
        self.linear = nn.Linear(input_dim, m)  # Output layer matches manifold dimension
       
        # Learnable residual connection scaling
        self.dt = nn.Parameter(torch.full((nlayers,), float(dt)))

    def forward(self, x: torch.Tensor, return_intermediate: bool = False) -> torch.Tensor:
        """
        Forward pass through the transformer with projections.
        
        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, input_size]
        return_intermediate : bool, default=False
            If True, returns intermediate outputs after each layer.
            
        Returns
        -------
        output : torch.Tensor
            Output tensor of shape [batch_size, seq_len, input_size]
            If return_intermediate=True, returns (intermediates, output) tuple
        """
       
        # Add positional encoding (transformer expects [seq_len, batch_size, d_model])
        g = x
        
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, d_model]

        intermediates = []
        # Process through transformer layers with optional internal projections
        for i in range(self.nlayers):
            x_in = x
            x = self.transformer_encoder[i](x)  # [seq_len, batch_size, d_model]
            x = torch.nn.functional.relu(x)
            
            # Apply internal projection if enabled
            if self.use_internal_exp:
                if self.internal_exp_func is not None:
                    x = self.internal_exp_func(x_in, x[:,:,:self.m], self.dt[i])
            
            if return_intermediate:
                intermediates.append(x.clone())
        
        # Convert to [batch_size, seq_len, d_model] for output layer
        x = x.permute(1, 0, 2)
        
        # Final output layer: [batch_size, seq_len, d_model] -> [batch_size, seq_len, input_size]
        output = self.linear(x)
        
        # ALWAYS apply end projection (cannot be disabled)
        if self.end_exp_func is not None:
            if self.use_internal_exp:
                output = self.end_exp_func(x, output, 1)
            else:
                output = self.end_exp_func(g, output, 1)
        
        if return_intermediate:
            return intermediates, output
        return output

class FFBlock(nn.Module):
    """One FF block: d_model -> d_hid -> d_model."""
    def __init__(self, d_model: int, d_hid: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hid)
        self.act = torch.nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hid, d_hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

class ExponentialFeedForward(nn.Module):
    """
    Stacked FF network with nlayers blocks.
    Output shape matches input shape: [..., d_model].
    """
    def __init__(
        self,
        d_model: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
        exp_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_internal_exponential: bool = True,
        dt: float = 1.0,   # scale each residual update
    ):
        super().__init__()
        if nlayers < 1:
            raise ValueError("nlayers must be >= 1")
        if use_internal_exponential:
            self.blocks = nn.ModuleList([FFBlock(d_model, d_hid, dropout) for _ in range(nlayers)])
        else:
            layers = [FFBlock(d_model, d_model, dropout) for _ in range(nlayers-1)]
            layers.append(FFBlock(d_model, d_hid, dropout))
            self.blocks = nn.ModuleList(layers)
        self.use_internal_exponential = use_internal_exponential
        self.internal_exp_func = exp_func
        self.final_exp_func = exp_func        
        self.dt = nn.Parameter(torch.full((nlayers,), float(dt)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x
        for i, blk in enumerate(self.blocks):
            y = blk(x)
            # Internal projection BETWEEN blocks (i.e., after block i, before block i+1)
            if i < len(self.blocks) - 1:
                if self.use_internal_exponential and self.internal_exp_func is not None:
                    x = self.internal_exp_func(x, y, self.dt[i])
                else:
                    x = y
            else:
                if self.final_exp_func is not None:
                    if self.use_internal_exponential:
                        x = self.final_exp_func(x, y, self.dt[i])  
                    else:
                        x = self.final_exp_func(g, y, self.dt[i])  
                else:
                    x = y
        return x

