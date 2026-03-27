# pose/losses/smooth_wing.py
"""
Smooth-Wing Loss for facial landmark localization.

Implementation based on:
"LOTR: Face Landmark Localization Using Localization Transformer"
Watchareeruetai et al., IEEE Access, 2022

The smooth-Wing loss addresses gradient discontinuity issues in the
original Wing loss while maintaining its key property of paying more
attention to small errors.
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from pose.registry import register_loss


@register_loss("wing")
class WingLoss(nn.Module):
    """
    Original Wing Loss for facial landmark localization.
    
    From: "Wing Loss for Robust Facial Landmark Localisation with 
    Convolutional Neural Networks" (Feng et al., CVPR 2018)
    
    L(x) = w * ln(1 + |x|/eps)  if |x| < w
           |x| - c              otherwise
           
    where c = w - w * ln(1 + w/eps)
    
    Args:
        w: Threshold parameter (typical values: 10-15)
        eps: Controls steepness of log curve (typical: 2.0)
    """
    
    def __init__(self, w: float = 10.0, eps: float = 2.0):
        super().__init__()
        self.w = w
        self.eps = eps
        self.c = w - w * math.log(1 + w / eps)
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Wing loss between predicted and target landmarks.
        
        Args:
            pred: Predicted landmarks (B, N, 2) or (B, N, 3)
            target: Ground truth landmarks (B, N, 2) or (B, N, 3)
            visible: Visibility mask (B, N) - optional
            sample_weight: Per-sample weights (B,) - optional
            
        Returns:
            Scalar loss value
        """
        # Compute per-coordinate error
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # Apply Wing loss formula
        # Small errors: log-based
        log_term = self.w * torch.log(1 + abs_diff / self.eps)
        # Large errors: linear
        linear_term = abs_diff - self.c
        
        # Combine based on threshold
        loss = torch.where(abs_diff < self.w, log_term, linear_term)
        
        # Sum over coordinates (x, y, [z])
        loss = loss.sum(dim=-1)  # (B, N)
        
        # Apply visibility mask
        if visible is not None:
            loss = loss * visible
            # Mean over visible landmarks only
            num_visible = visible.sum(dim=1).clamp(min=1)
            loss = loss.sum(dim=1) / num_visible  # (B,)
        else:
            loss = loss.mean(dim=1)  # (B,)
            
        # Apply sample weights
        if sample_weight is not None:
            loss = loss * sample_weight
            return loss.sum() / sample_weight.sum().clamp(min=1e-6)
        else:
            return loss.mean()


@register_loss("smooth_wing")
class SmoothWingLoss(nn.Module):
    """
    Smooth-Wing Loss for facial landmark localization.
    
    Modification of Wing Loss that eliminates gradient discontinuities
    at zero error and at the threshold w.
    
    Key improvements over Wing Loss:
    1. Behaves like L2 loss for very small errors (|x| < t)
       - Gradient is smooth at x=0
    2. Smooth transition at the outer threshold w
       - No abrupt gradient change
    
    Loss function:
        L(x) = s * |x|^2                           if |x| < t
               w * ln(1 + |x|/eps) + c1            if t <= |x| < w  
               |x| - c2                            if |x| >= w
               
    where:
        t: inner threshold (small error region uses L2)
        w: outer threshold (beyond w, use linear)
        s, c1, c2: constants for smooth transitions
    
    Args:
        w: Outer threshold (typical: 10.0)
        eps: Steepness parameter for log (typical: 2.0)
        t: Inner threshold for L2 region (typical: 2.0, must be < w)
    """
    
    def __init__(
        self, 
        w: float = 10.0, 
        eps: float = 2.0, 
        t: float = 2.0,
        use_uncertainty: bool = False,
        log_var_min: float = -6.0,
        log_var_max: float = 6.0,
        confidence_supervision_weight: float = 0.05,
        confidence_target_sigma: float = 0.05,
        lambda_conf: Optional[float] = None,
        sigma: Optional[float] = None,
    ):
        super().__init__()
        assert 0 < t < w, f"Inner threshold t must be in (0, w), got t={t}, w={w}"
        
        self.w = w
        self.eps = eps
        self.t = t
        self.use_uncertainty = bool(use_uncertainty)
        self.log_var_min = float(log_var_min)
        self.log_var_max = float(log_var_max)
        # Backward/forward-compatible aliases for config friendliness.
        if lambda_conf is not None:
            confidence_supervision_weight = float(lambda_conf)
        if sigma is not None:
            confidence_target_sigma = float(sigma)
        self.confidence_supervision_weight = float(confidence_supervision_weight)
        self.confidence_target_sigma = float(confidence_target_sigma)
        
        # Compute smoothing constants for continuity
        # At x = t: s*t^2 = w*ln(1 + t/eps) + c1
        # Derivative at x = t: 2*s*t = w/(eps + t)
        # => s = w / (2*t*(eps + t))
        self.s = w / (2.0 * t * (eps + t))
        
        # c1 ensures continuity at t
        self.c1 = self.s * t * t - w * math.log(1 + t / eps)
        
        # c2 ensures continuity at w
        # w*ln(1 + w/eps) + c1 = w - c2
        # => c2 = w - w*ln(1 + w/eps) - c1
        self.c2 = w - w * math.log(1 + w / eps) - self.c1
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        pred_log_var: Optional[torch.Tensor] = None,
        pred_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Smooth-Wing loss between predicted and target landmarks.
        
        Args:
            pred: Predicted landmarks (B, N, 2) or (B, N, 3)
            target: Ground truth landmarks (B, N, 2) or (B, N, 3)
            visible: Visibility mask (B, N) - optional
            sample_weight: Per-sample weights (B,) - optional
            
        Returns:
            Scalar loss value
        """
        diff = pred - target
        abs_diff = torch.abs(diff)
        # Per-landmark geometric error and a per-sample normalization scale.
        point_err = torch.linalg.norm(diff, dim=-1)  # (B, N)

        # Normalize error by face bbox diagonal in target-landmark space.
        # This keeps confidence targets scale-consistent across face sizes.
        x_t = target[..., 0]
        y_t = target[..., 1]
        if visible is not None:
            vis = visible > 0
            x_min = torch.where(vis, x_t, torch.full_like(x_t, float("inf"))).min(dim=1).values
            x_max = torch.where(vis, x_t, torch.full_like(x_t, float("-inf"))).max(dim=1).values
            y_min = torch.where(vis, y_t, torch.full_like(y_t, float("inf"))).min(dim=1).values
            y_max = torch.where(vis, y_t, torch.full_like(y_t, float("-inf"))).max(dim=1).values
            no_vis = ~(vis.any(dim=1))
            if no_vis.any():
                x_min_all = x_t.min(dim=1).values
                x_max_all = x_t.max(dim=1).values
                y_min_all = y_t.min(dim=1).values
                y_max_all = y_t.max(dim=1).values
                x_min = torch.where(no_vis, x_min_all, x_min)
                x_max = torch.where(no_vis, x_max_all, x_max)
                y_min = torch.where(no_vis, y_min_all, y_min)
                y_max = torch.where(no_vis, y_max_all, y_max)
        else:
            x_min = x_t.min(dim=1).values
            x_max = x_t.max(dim=1).values
            y_min = y_t.min(dim=1).values
            y_max = y_t.max(dim=1).values

        face_diag = torch.sqrt((x_max - x_min).clamp(min=1e-6) ** 2 + (y_max - y_min).clamp(min=1e-6) ** 2)
        norm_error = point_err / face_diag.unsqueeze(1)
        
        # Three regions:
        # 1. |x| < t: L2-like region
        l2_term = self.s * abs_diff * abs_diff
        
        # 2. t <= |x| < w: Wing loss region
        wing_term = self.w * torch.log(1 + abs_diff / self.eps) + self.c1
        
        # 3. |x| >= w: Linear region
        linear_term = abs_diff - self.c2
        
        # Combine regions
        loss = torch.where(
            abs_diff < self.t,
            l2_term,
            torch.where(abs_diff < self.w, wing_term, linear_term)
        )
        
        # Sum over coordinates
        loss = loss.sum(dim=-1)  # (B, N)

        # Optional heteroscedastic uncertainty weighting.
        # NLL form: exp(-s) * L + s, where s = log variance.
        if self.use_uncertainty and pred_log_var is not None:
            s = pred_log_var
            if s.dim() == 3 and s.shape[-1] == 1:
                s = s.squeeze(-1)
            if s.shape != loss.shape:
                raise ValueError(
                    f"pred_log_var must have shape {tuple(loss.shape)} (or [...,1]), got {tuple(s.shape)}"
                )
            s = torch.clamp(s, min=self.log_var_min, max=self.log_var_max)
            loss = torch.exp(-s) * loss + s

        conf_loss = None
        if self.confidence_supervision_weight > 0.0:
            sig = max(self.confidence_target_sigma, 1e-6)
            # Detach target so confidence branch does not backprop through target construction.
            target_conf = torch.exp(-(norm_error.detach() ** 2) / (2.0 * sig * sig)).clamp(0.0, 1.0)
            if pred_confidence is not None:
                pred_conf = pred_confidence
                if pred_conf.dim() == 3 and pred_conf.shape[-1] == 1:
                    pred_conf = pred_conf.squeeze(-1)
                if pred_conf.shape != target_conf.shape:
                    raise ValueError(
                        f"pred_confidence must have shape {tuple(target_conf.shape)} (or [...,1]), got {tuple(pred_conf.shape)}"
                    )
                pred_conf = pred_conf.clamp(0.0, 1.0)
            elif self.use_uncertainty and pred_log_var is not None:
                pred_conf = torch.sigmoid(-s)
            else:
                pred_conf = None
            if pred_conf is not None:
                conf_loss = (pred_conf - target_conf) ** 2  # (B, N)
        
        # Apply visibility mask
        if visible is not None:
            loss = loss * visible
            num_visible = visible.sum(dim=1).clamp(min=1)
            loss = loss.sum(dim=1) / num_visible
            if conf_loss is not None:
                conf_loss = conf_loss * visible
                conf_loss = conf_loss.sum(dim=1) / num_visible
        else:
            loss = loss.mean(dim=1)
            if conf_loss is not None:
                conf_loss = conf_loss.mean(dim=1)

        if conf_loss is not None:
            loss = loss + self.confidence_supervision_weight * conf_loss
            
        # Apply sample weights
        if sample_weight is not None:
            loss = loss * sample_weight
            return loss.sum() / sample_weight.sum().clamp(min=1e-6)
        else:
            return loss.mean()


@register_loss("adaptive_wing")
class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for facial landmark localization.
    
    From: "Adaptive Wing Loss for Robust Face Alignment via Heatmap 
    Regression" (Wang et al., ICCV 2019)
    
    An adaptive variant that automatically adjusts loss behavior based
    on error magnitude, with better gradients for small errors.
    
    L(x) = w * ln(1 + |x/eps|^(alpha-y))  if |x| < theta
           A|x| - C                        otherwise
           
    Args:
        theta: Threshold (typical: 0.5)
        alpha: Power parameter (typical: 2.1)
        omega: Balance weight (typical: 14.0)
        eps: Small constant for stability (typical: 1.0)
    """
    
    def __init__(
        self, 
        theta: float = 0.5,
        alpha: float = 2.1,
        omega: float = 14.0,
        eps: float = 1.0
    ):
        super().__init__()
        self.theta = theta
        self.alpha = alpha
        self.omega = omega
        self.eps = eps
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Adaptive Wing loss.
        
        Args:
            pred: Predicted landmarks (B, N, 2)
            target: Ground truth landmarks (B, N, 2)
            visible: Visibility mask (B, N) - optional
            sample_weight: Per-sample weights (B,) - optional
        """
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # Adaptive parameters
        y = target.detach()  # Use target values to adapt
        
        # Compute loss for small errors
        A = self.omega * (
            1.0 / (1.0 + torch.pow(self.theta / self.eps, self.alpha - y))
        ) * (self.alpha - y) * torch.pow(self.theta / self.eps, self.alpha - y - 1) / self.eps
        
        C = self.theta * A - self.omega * torch.log(
            1.0 + torch.pow(self.theta / self.eps, self.alpha - y)
        )
        
        # Two regions
        small_error = self.omega * torch.log(
            1.0 + torch.pow(abs_diff / self.eps, self.alpha - y)
        )
        large_error = A * abs_diff - C
        
        loss = torch.where(abs_diff < self.theta, small_error, large_error)
        
        # Sum over coordinates
        loss = loss.sum(dim=-1)
        
        # Apply visibility mask
        if visible is not None:
            loss = loss * visible
            num_visible = visible.sum(dim=1).clamp(min=1)
            loss = loss.sum(dim=1) / num_visible
        else:
            loss = loss.mean(dim=1)
            
        # Apply sample weights
        if sample_weight is not None:
            loss = loss * sample_weight
            return loss.sum() / sample_weight.sum().clamp(min=1e-6)
        else:
            return loss.mean()


@register_loss("landmark_l1")
class LandmarkL1Loss(nn.Module):
    """
    L1 Loss for landmark coordinate regression.
    Simple but effective baseline for landmark localization.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute L1 loss between predicted and target landmarks.
        """
        diff = torch.abs(pred - target)
        loss = diff.sum(dim=-1)  # (B, N)
        
        if visible is not None:
            loss = loss * visible
            num_visible = visible.sum(dim=1).clamp(min=1)
            loss = loss.sum(dim=1) / num_visible
        else:
            loss = loss.mean(dim=1)
            
        if sample_weight is not None:
            loss = loss * sample_weight
            return loss.sum() / sample_weight.sum().clamp(min=1e-6)
        else:
            return loss.mean()


@register_loss("landmark_mse")
class LandmarkMSELoss(nn.Module):
    """
    MSE (L2) Loss for landmark coordinate regression.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target landmarks.
        """
        diff_sq = (pred - target) ** 2
        loss = diff_sq.sum(dim=-1)  # (B, N)
        
        if visible is not None:
            loss = loss * visible
            num_visible = visible.sum(dim=1).clamp(min=1)
            loss = loss.sum(dim=1) / num_visible
        else:
            loss = loss.mean(dim=1)
            
        if sample_weight is not None:
            loss = loss * sample_weight
            return loss.sum() / sample_weight.sum().clamp(min=1e-6)
        else:
            return loss.mean()


@register_loss("landmark_smooth_l1")
class LandmarkSmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) Loss for landmark coordinate regression.
    Combines benefits of L1 (robustness) and L2 (smooth gradients).
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Smooth L1 loss between predicted and target landmarks.
        """
        diff = torch.abs(pred - target)
        
        # Smooth L1 formula
        loss = torch.where(
            diff < self.beta,
            0.5 * diff * diff / self.beta,
            diff - 0.5 * self.beta
        )
        
        loss = loss.sum(dim=-1)  # (B, N)
        
        if visible is not None:
            loss = loss * visible
            num_visible = visible.sum(dim=1).clamp(min=1)
            loss = loss.sum(dim=1) / num_visible
        else:
            loss = loss.mean(dim=1)
            
        if sample_weight is not None:
            loss = loss * sample_weight
            return loss.sum() / sample_weight.sum().clamp(min=1e-6)
        else:
            return loss.mean()


@register_loss("nme")
class NMELoss(nn.Module):
    """
    Normalized Mean Error (NME) Loss.
    
    Normalizes the error by face size (inter-pupil distance or 
    bounding box diagonal) for scale-invariant training.
    
    Args:
        normalization: Type of normalization
            - 'interocular': Use inter-pupil distance
            - 'bbox': Use bounding box diagonal
            - 'face_size': Use explicit face size if provided
        left_eye_indices: Indices of left eye landmarks for interocular
        right_eye_indices: Indices of right eye landmarks for interocular
    """
    
    def __init__(
        self,
        normalization: str = "interocular",
        left_eye_indices: Optional[list] = None,
        right_eye_indices: Optional[list] = None,
        smooth: bool = True
    ):
        super().__init__()
        self.normalization = normalization
        # Default to 300W eye indices (36-41 left eye, 42-47 right eye)
        self.left_eye_indices = left_eye_indices or [36, 37, 38, 39, 40, 41]
        self.right_eye_indices = right_eye_indices or [42, 43, 44, 45, 46, 47]
        self.smooth = smooth
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        face_size: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute NME loss.
        
        Args:
            pred: Predicted landmarks (B, N, 2)
            target: Ground truth landmarks (B, N, 2)
            visible: Visibility mask (B, N)
            sample_weight: Per-sample weights (B,)
            face_size: Explicit normalization factor (B,) - optional
        """
        B, N, _ = pred.shape
        device = pred.device
        
        # Compute normalization factor
        if face_size is not None:
            norm_factor = face_size
        elif self.normalization == "interocular":
            # Compute inter-ocular distance from target
            left_eye = target[:, self.left_eye_indices, :2].mean(dim=1)  # (B, 2)
            right_eye = target[:, self.right_eye_indices, :2].mean(dim=1)  # (B, 2)
            norm_factor = torch.norm(left_eye - right_eye, dim=1)  # (B,)
        elif self.normalization == "bbox":
            # Use bounding box diagonal
            x_min = target[..., 0].min(dim=1).values
            x_max = target[..., 0].max(dim=1).values
            y_min = target[..., 1].min(dim=1).values
            y_max = target[..., 1].max(dim=1).values
            norm_factor = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        else:
            # No normalization
            norm_factor = torch.ones(B, device=device)
            
        norm_factor = norm_factor.clamp(min=1e-6).unsqueeze(1)  # (B, 1)
        
        # Compute per-landmark error
        diff = torch.norm(pred[..., :2] - target[..., :2], dim=-1)  # (B, N)
        
        # Normalize by face size
        normalized_error = diff / norm_factor  # (B, N)
        
        # Apply visibility mask
        if visible is not None:
            normalized_error = normalized_error * visible
            num_visible = visible.sum(dim=1).clamp(min=1)
            mean_error = normalized_error.sum(dim=1) / num_visible
        else:
            mean_error = normalized_error.mean(dim=1)
            
        # Use smooth L1-like gradient for stable training
        if self.smooth:
            threshold = 0.1  # 10% of face size
            loss = torch.where(
                mean_error < threshold,
                0.5 * mean_error * mean_error / threshold,
                mean_error - 0.5 * threshold
            )
        else:
            loss = mean_error
            
        # Apply sample weights
        if sample_weight is not None:
            loss = loss * sample_weight
            return loss.sum() / sample_weight.sum().clamp(min=1e-6)
        else:
            return loss.mean()


@register_loss("combined_landmark")
class CombinedLandmarkLoss(nn.Module):
    """
    Combined loss that uses multiple landmark loss functions.
    
    Useful for balancing different objectives during training.
    
    Args:
        losses: Dict mapping loss names to weights
            e.g., {"smooth_wing": 1.0, "nme": 0.5}
    """
    
    def __init__(
        self,
        losses: dict = None,
        **kwargs
    ):
        super().__init__()
        
        if losses is None:
            losses = {"smooth_wing": 1.0}
            
        self.loss_weights = {}
        self.loss_modules = nn.ModuleDict()
        
        from pose.registry import LOSS_REGISTRY
        
        for name, weight in losses.items():
            if name in LOSS_REGISTRY:
                self.loss_modules[name] = LOSS_REGISTRY[name]()
                self.loss_weights[name] = weight
            else:
                raise ValueError(f"Unknown loss: {name}")
                
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        visible: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute weighted sum of losses."""
        total_loss = 0.0
        
        for name, module in self.loss_modules.items():
            weight = self.loss_weights[name]
            loss = module(pred, target, visible, sample_weight, **kwargs)
            total_loss = total_loss + weight * loss
            
        return total_loss
