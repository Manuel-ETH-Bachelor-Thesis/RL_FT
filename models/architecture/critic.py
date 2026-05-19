"""
Critic networks.
Adapted from
@inproceedings{dppo2024,
    title={Diffusion Policy Policy Optimization},
    author={Ren, Allen Z. and Lidard, Justin and Ankile, Lars L. and Simeonov, Anthony and Agrawal, Pulkit and Majumdar, Anirudha and Burchfiel, Benjamin and Dai, Hongkai and Simchowitz, Max},
    booktitle={arXiv preprint arXiv:2409.00588},
    year={2024}
}
"""

from typing import Union
import torch
import einops
from copy import deepcopy

from model.common.mlp import MLP, ResidualMLP
from model.common.modules import SpatialEmb, RandomShiftsAug


class CriticObs(torch.nn.Module):
    """State-only critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim] + mlp_dims + [1]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, cond: Union[dict, torch.Tensor]):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            or (B, num_feature) from ViT encoder
        """
        if isinstance(cond, dict):
            B = len(cond["state"])

            # flatten history
            state = cond["state"].view(B, -1)
        else:
            state = cond
        q1 = self.Q1(state)
        return q1

class CriticObs_Mod(torch.nn.Module):
    """State-only critic network. (Modified for proprioception test run)"""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim] + mlp_dims + [1]
        
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
            
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, cond: Union[dict, torch.Tensor]):
        """
        cond: dict with keys "proprio", "rgb", "depth"
            proprio: (B, cond_steps, obs_dim)
        """
        if isinstance(cond, dict):
            B = len(cond["proprio"])
            
            state = cond["proprio"].reshape(B, -1)
        else:
            state = cond
            
        q1 = self.Q1(state)
        return q1


class CriticObsAct(torch.nn.Module):
    """State-action double critic network."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        action_dim,
        action_steps=1,
        activation_type="Mish",
        use_layernorm=False,
        residual_tyle=False,
        double_q=True,
        **kwargs,
    ):
        super().__init__()
        mlp_dims = [cond_dim + action_dim * action_steps] + mlp_dims + [1]
        if residual_tyle:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )
        if double_q:
            self.Q2 = model(
                mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
                use_layernorm=use_layernorm,
            )

    def forward(self, cond: dict, action):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        action: (B, Ta, Da)
        """
        B = len(cond["state"])

        # flatten history
        state = cond["state"].view(B, -1)

        # flatten action
        action = action.view(B, -1)

        x = torch.cat((state, action), dim=-1)
        if hasattr(self, "Q2"):
            q1 = self.Q1(x)
            q2 = self.Q2(x)
            return q1.squeeze(1), q2.squeeze(1)
        else:
            q1 = self.Q1(x)
            return q1.squeeze(1)


class ViTCritic(CriticObs):
    """ViT + MLP, state only"""

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        # update input dim to mlp
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.backbone = backbone
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:  # TODO: clean up
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def forward(
        self,
        cond: dict,
        no_augment=False,
    ):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        no_augment: whether to skip augmentation

        TODO long term: more flexible handling of cond
        """
        B, T_rgb, C, H, W = cond["rgb"].shape

        # flatten history
        state = cond["state"].view(B, -1)

        # Take recent images --- sometimes we want to use fewer img_cond_steps than cond_steps (e.g., 1 image but 3 prio)
        rgb = cond["rgb"][:, -self.img_cond_steps :]

        # concatenate images in cond by channels
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        # convert rgb to float32 for augmentation
        rgb = rgb.float()

        # get vit output - pass in two images separately
        if self.num_img > 1:  # TODO: properly handle multiple images
            rgb1 = rgb[:, 0]
            rgb2 = rgb[:, 1]
            if self.augment and not no_augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.backbone(rgb1)
            feat2 = self.backbone(rgb2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  # single image
            if self.augment and not no_augment:
                rgb = self.aug(rgb)  # uint8 -> float32
            feat = self.backbone(rgb)
            feat = self.compress.forward(feat, state)
        feat = torch.cat([feat, state], dim=-1)
        return super().forward(feat)


class ViTCritic_Mod(CriticObs):
    """ViT + MLP, state only. Modified for proprio, rgb, and depth."""

    def __init__(
        self,
        backbone,
        cond_dim,
        img_cond_steps=1,
        spatial_emb=128,
        dropout=0,
        augment=False,
        num_img=1,
        **kwargs,
    ):
        mlp_obs_dim = spatial_emb * num_img + cond_dim
        super().__init__(cond_dim=mlp_obs_dim, **kwargs)
        self.backbone = backbone
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps
        
        if num_img > 1:
            self.compress1 = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            self.compress2 = deepcopy(self.compress1)
        else:
            self.compress = SpatialEmb(
                num_patch=self.backbone.num_patch,
                patch_dim=self.backbone.patch_repr_dim,
                prop_dim=cond_dim,
                proj_dim=spatial_emb,
                dropout=dropout,
            )
            
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment

    def forward(self, cond: dict, no_augment=False):
        """
        cond: dict with keys proprio, rgb, depth.
            proprio: (B, To, Do)
            rgb: (B, To, 3, H, W)
            depth: (B, To, 1, H, W)
        """
        B, T_rgb, C_rgb, H, W = cond["rgb"].shape

        state = cond["proprio"].view(B, -1)

        rgb = cond["rgb"][:, -self.img_cond_steps:]
        depth = cond["depth"][:, -self.img_cond_steps:]

        rgbd = torch.cat([rgb, depth], dim=2) 

        if self.num_img > 1:
            rgbd = rgbd.reshape(B, T_rgb, self.num_img, 4, H, W)
            rgbd = einops.rearrange(rgbd, "b t n c h w -> b n (t c) h w")
        else:
            rgbd = einops.rearrange(rgbd, "b t c h w -> b (t c) h w")

        rgbd = rgbd.float()

        if self.num_img > 1:
            rgbd1 = rgbd[:, 0]
            rgbd2 = rgbd[:, 1]
            if self.augment and not no_augment:
                rgbd1 = self.aug(rgbd1)
                rgbd2 = self.aug(rgbd2)
            feat1 = self.backbone(rgbd1)
            feat2 = self.backbone(rgbd2)
            feat1 = self.compress1.forward(feat1, state)
            feat2 = self.compress2.forward(feat2, state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:  
            if self.augment and not no_augment:
                rgbd = self.aug(rgbd)
            feat = self.backbone(rgbd)
            feat = self.compress.forward(feat, state)
            
        feat = torch.cat([feat, state], dim=-1)
        
        return super().forward(feat)