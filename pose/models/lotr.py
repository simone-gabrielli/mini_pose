# pose/models/lotr.py
"""
LOTR: Localization Transformer for Facial Landmark Detection

Implementation based on:
"LOTR: Face Landmark Localization Using Localization Transformer"
Watchareeruetai et al., IEEE Access, 2022

Key features:
- Visual backbone (MobileNetV2, ResNet50, or HRNet)
- Transformer encoder-decoder architecture
- Direct coordinate regression (no heatmaps)
- 2D positional encoding for spatial tokens
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from torchvision.models import mobilenet_v2, resnet50, MobileNet_V2_Weights, ResNet50_Weights

from pose.models.base import PoseModel
from pose.registry import register_model


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for feature map tokens.
    
    Adds sinusoidal positional encoding to tokens based on their 
    spatial position in the original feature map.
    """
    
    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        
        # Pre-compute positional encodings
        pe = torch.zeros(max_h, max_w, d_model)
        
        # Create position indices
        y_pos = torch.arange(0, max_h).unsqueeze(1).repeat(1, max_w)
        x_pos = torch.arange(0, max_w).unsqueeze(0).repeat(max_h, 1)
        
        # Compute div term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * 
                            -(math.log(temperature) / (d_model // 2)))
        
        # X dimension encoding (even indices)
        pe[:, :, 0::4] = torch.sin(x_pos.unsqueeze(-1).float() * div_term)
        pe[:, :, 1::4] = torch.cos(x_pos.unsqueeze(-1).float() * div_term)
        
        # Y dimension encoding (odd indices)  
        pe[:, :, 2::4] = torch.sin(y_pos.unsqueeze(-1).float() * div_term)
        pe[:, :, 3::4] = torch.cos(y_pos.unsqueeze(-1).float() * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, h: int, w: int) -> torch.Tensor:
        """Returns positional encoding of shape (h*w, d_model)"""
        return self.pe[:h, :w, :].reshape(-1, self.d_model)


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer encoder layer with pre-norm."""
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(
        self, 
        src: torch.Tensor, 
        pos: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm self-attention with positional encoding added to Q and K
        q = k = src if pos is None else src + pos
        src2, _ = self.self_attn(q, k, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention, cross-attention, and FFN."""
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        # Self-attention on landmark queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross-attention to encoder output
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention on queries
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention to encoder memory
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if pos is None else memory + pos
        tgt2, _ = self.cross_attn(q, k, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                encoder_layer.self_attn.embed_dim,
                encoder_layer.self_attn.num_heads,
                encoder_layer.linear1.out_features,
                encoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
    def forward(
        self, 
        src: torch.Tensor, 
        pos: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos, src_key_padding_mask=src_key_padding_mask)
        return output


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""
    
    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                decoder_layer.self_attn.embed_dim,
                decoder_layer.self_attn.num_heads,
                decoder_layer.linear1.out_features,
                decoder_layer.dropout.p
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos,
                          memory_key_padding_mask=memory_key_padding_mask)
        return output


class LandmarkPredictionHead(nn.Module):
    """MLP head for predicting landmark coordinates from decoder output."""
    
    def __init__(
        self, 
        d_model: int, 
        hidden_dim: int = 512,
        num_hidden_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        in_dim = d_model
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
            
        # Final output layer (no activation - predicts normalized coords)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize final layer with small weights for stable training
        nn.init.xavier_uniform_(self.mlp[-1].weight, gain=0.01)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) decoder output
        Returns:
            coords: (B, N, 2) normalized coordinates in [0, 1]
        """
        coords = self.mlp(x)
        # Apply sigmoid to constrain to [0, 1] range
        coords = torch.sigmoid(coords)
        return coords


class DeconvUpsampler(nn.Module):
    """Deconvolution-based upsampling to increase feature map resolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        kernel_size: int = 4
    ):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            # Reduce channels gradually
            next_channels = out_channels if i == num_layers - 1 else current_channels // 2
            
            layers.append(nn.ConvTranspose2d(
                current_channels, next_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2 - 1,
                output_padding=0
            ))
            layers.append(nn.BatchNorm2d(next_channels))
            layers.append(nn.ReLU(inplace=True))
            
            current_channels = next_channels
            
        self.deconv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


@register_model("lotr")
class LOTR(PoseModel):
    """
    Localization Transformer (LOTR) for facial landmark detection.
    
    Architecture:
        1. Visual Backbone (MobileNetV2/ResNet50) -> Feature map
        2. Optional deconvolution upsampling
        3. 1x1 Conv to reduce channels to d_model
        4. Transformer Encoder (spatial features -> enriched features)
        5. Transformer Decoder (landmark queries + encoder output -> landmark features)
        6. Prediction Head (MLP -> 2D coordinates)
    
    Args:
        num_keypoints: Number of landmarks to predict
        backbone: Backbone type ('mobilenet_v2', 'resnet50')
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer encoder layers
        num_decoder_layers: Number of transformer decoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout rate
        use_upsampling: Whether to use deconv upsampling
        upsampling_layers: Number of deconv layers for upsampling
        pretrained: Whether to use pretrained backbone
        input_size: Expected input size (H, W) for positional encoding
    """
    
    def __init__(
        self,
        num_keypoints: int,
        backbone: str = "mobilenet_v2",
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_upsampling: bool = True,
        upsampling_layers: int = 2,
        pretrained: bool = True,
        input_size: Tuple[int, int] = (256, 256),
        prediction_output_dim: int = 2,  # 2 for 2D, 3 for 3D landmarks
        **kwargs  # Accept extra kwargs for compatibility
    ):
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.input_size = input_size
        self.prediction_output_dim = prediction_output_dim
        
        # Build backbone
        self.backbone, backbone_channels = self._build_backbone(backbone, pretrained)
        
        # Calculate feature map size after backbone
        # MobileNetV2 and ResNet50 produce stride-32 features
        self.feature_h = input_size[0] // 32
        self.feature_w = input_size[1] // 32
        
        # Optional upsampling to increase spatial resolution
        self.use_upsampling = use_upsampling
        if use_upsampling:
            self.upsampler = DeconvUpsampler(
                in_channels=backbone_channels,
                out_channels=d_model,
                num_layers=upsampling_layers
            )
            # Update feature size after upsampling
            self.feature_h *= (2 ** upsampling_layers)
            self.feature_w *= (2 ** upsampling_layers)
            self.input_proj = nn.Identity()  # Already at d_model channels
        else:
            # 1x1 conv to reduce channels to d_model
            self.input_proj = nn.Conv2d(backbone_channels, d_model, kernel_size=1)
            self.upsampler = None
            
        # 2D Positional encoding for spatial tokens
        self.pos_encoding = PositionalEncoding2D(
            d_model, 
            max_h=self.feature_h, 
            max_w=self.feature_w
        )
        
        # Learnable landmark queries (one per landmark)
        self.landmark_queries = nn.Embedding(num_keypoints, d_model)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Prediction head
        self.prediction_head = LandmarkPredictionHead(
            d_model=d_model,
            hidden_dim=dim_feedforward,
            num_hidden_layers=2,
            output_dim=prediction_output_dim,
            dropout=dropout
        )
        
        self._init_weights()
        
    def _build_backbone(self, backbone: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Build the visual backbone network."""
        
        if backbone == "mobilenet_v2":
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            model = mobilenet_v2(weights=weights)
            # Use features up to stride 32
            backbone_net = model.features
            out_channels = 1280
            
        elif backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet50(weights=weights)
            # Remove avgpool and fc, keep conv layers
            layers = [
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4
            ]
            backbone_net = nn.Sequential(*layers)
            out_channels = 2048
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Supported: 'mobilenet_v2', 'resnet50'")
            
        return backbone_net, out_channels
    
    def _init_weights(self):
        """Initialize transformer and projection weights."""
        # Initialize landmark queries
        nn.init.normal_(self.landmark_queries.weight, std=0.01)
        
        # Initialize projection layer if exists
        if hasattr(self.input_proj, 'weight'):
            nn.init.xavier_uniform_(self.input_proj.weight)
            if self.input_proj.bias is not None:
                nn.init.zeros_(self.input_proj.bias)
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            
        Returns:
            landmarks: Predicted landmark coordinates (B, N, 2) in normalized [0, 1] range
            
            For compatibility with trainer:
            - landmarks_pixel: (B, N, 2) coordinates scaled to image size
        """
        B = x.size(0)
        
        # 1. Extract visual features
        features = self.backbone(x)  # (B, C, H/32, W/32)
        
        # 2. Optional upsampling
        if self.use_upsampling and self.upsampler is not None:
            features = self.upsampler(features)  # (B, d_model, H', W')
        else:
            features = self.input_proj(features)  # (B, d_model, H', W')
        
        _, _, H_feat, W_feat = features.shape
        
        # 3. Flatten spatial dimensions to sequence
        # (B, d_model, H, W) -> (B, H*W, d_model)
        features_seq = features.flatten(2).permute(0, 2, 1)
        
        # 4. Get positional encoding for current feature map size
        pos_enc = self.pos_encoding(H_feat, W_feat)  # (H*W, d_model)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)  # (B, H*W, d_model)
        
        # 5. Transformer encoder
        memory = self.encoder(features_seq, pos=pos_enc)  # (B, H*W, d_model)
        
        # 6. Prepare landmark queries
        query_embed = self.landmark_queries.weight  # (N, d_model)
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_model)
        
        # Initial query (zeros, will be refined by decoder)
        tgt = torch.zeros_like(query_embed)
        
        # 7. Transformer decoder
        decoder_out = self.decoder(
            tgt, 
            memory, 
            pos=pos_enc,
            query_pos=query_embed
        )  # (B, N, d_model)
        
        # 8. Predict coordinates
        landmarks_norm = self.prediction_head(decoder_out)  # (B, N, 2)
        
        # Scale to image size for compatibility with existing evaluation
        H_img, W_img = self.input_size
        landmarks_pixel = landmarks_norm.clone()
        landmarks_pixel[..., 0] *= W_img
        landmarks_pixel[..., 1] *= H_img
        
        return landmarks_norm, landmarks_pixel
    
    def get_landmarks_pixel(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method to get pixel coordinates directly."""
        _, landmarks_pixel = self.forward(x)
        return landmarks_pixel
    
    def generate_sample_visualization(
        self,
        sample: dict,
        out_path: str,
        device: torch.device,
    ) -> None:
        """Generate visualization for trainer's qualitative examples hook."""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        self.eval()
        img_tensor = sample["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, landmarks_pixel = self.forward(img_tensor)
        
        landmarks = landmarks_pixel[0].cpu().numpy()  # (N, 2)
        
        # Get image for display
        img_np = sample["image"].cpu().numpy().transpose(1, 2, 0)
        
        # Undo ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = np.clip((img_np * std) + mean, 0, 1)
        
        # Get ground truth keypoints
        keypts_gt = sample["keypoints"].cpu().numpy()
        
        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_np)
        
        # Plot GT landmarks in green
        ax.scatter(keypts_gt[:, 0], keypts_gt[:, 1], c='lime', s=8, 
                   label='GT', alpha=0.8, marker='o')
        
        # Plot predicted landmarks in red
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=8, 
                   label='Pred', alpha=0.8, marker='x')
        
        # Draw connections between GT and predictions for error visualization
        for i in range(len(landmarks)):
            ax.plot([keypts_gt[i, 0], landmarks[i, 0]], 
                    [keypts_gt[i, 1], landmarks[i, 1]], 
                    'b-', alpha=0.3, linewidth=0.5)
        
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('LOTR Landmark Predictions', fontsize=10)
        ax.axis('off')
        
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


@register_model("lotr_3d")
class LOTR3D(LOTR):
    """
    LOTR variant for 3D facial landmark detection.
    Predicts (x, y, z) coordinates for each landmark.
    """
    
    def __init__(
        self,
        num_keypoints: int,
        depth_range: Optional[Tuple[float, float]] = None,
        depth_mean: Optional[float] = None,
        **kwargs
    ):
        # Set output dimension to 3 for 3D landmarks
        kwargs['prediction_output_dim'] = 3
        super().__init__(num_keypoints=num_keypoints, **kwargs)
        
        self.depth_range = depth_range
        self.depth_mean = depth_mean
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for 3D landmarks.
        
        Returns:
            landmarks_norm: (B, N, 3) normalized coordinates
            landmarks_pixel: (B, N, 3) pixel coordinates (x, y scaled, z in depth units)
        """
        B = x.size(0)
        
        # Run through backbone and transformer
        features = self.backbone(x)
        
        if self.use_upsampling and self.upsampler is not None:
            features = self.upsampler(features)
        else:
            features = self.input_proj(features)
        
        _, _, H_feat, W_feat = features.shape
        features_seq = features.flatten(2).permute(0, 2, 1)
        
        pos_enc = self.pos_encoding(H_feat, W_feat)
        pos_enc = pos_enc.unsqueeze(0).expand(B, -1, -1)
        
        memory = self.encoder(features_seq, pos=pos_enc)
        
        query_embed = self.landmark_queries.weight.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query_embed)
        
        decoder_out = self.decoder(tgt, memory, pos=pos_enc, query_pos=query_embed)
        
        landmarks_norm = self.prediction_head(decoder_out)  # (B, N, 3)
        
        # Scale to image/depth space
        H_img, W_img = self.input_size
        landmarks_pixel = landmarks_norm.clone()
        landmarks_pixel[..., 0] *= W_img
        landmarks_pixel[..., 1] *= H_img
        
        # Scale z to depth range if provided
        if self.depth_range is not None:
            z_min, z_max = self.depth_range
            landmarks_pixel[..., 2] = landmarks_norm[..., 2] * (z_max - z_min) + z_min
        
        return landmarks_norm, landmarks_pixel


@register_model("lotr_light")
class LOTRLight(LOTR):
    """
    Lightweight LOTR variant for real-time applications.
    Uses smaller dimensions and fewer transformer layers.
    """
    
    def __init__(
        self,
        num_keypoints: int,
        **kwargs
    ):
        # Override defaults for lightweight model
        defaults = {
            'backbone': 'mobilenet_v2',
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dim_feedforward': 256,
            'use_upsampling': False,  # Skip upsampling for speed
        }
        defaults.update(kwargs)
        super().__init__(num_keypoints=num_keypoints, **defaults)
