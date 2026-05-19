"""
Policy network architecture for PPO Diffusion Agent.
Adapted from
@inproceedings{dppo2024,
    title={Diffusion Policy Policy Optimization},
    author={Ren, Allen Z. and Lidard, Justin and Ankile, Lars L. and Simeonov, Anthony and Agrawal, Pulkit and Majumdar, Anirudha and Burchfiel, Benjamin and Dai, Hongkai and Simchowitz, Max},
    booktitle={arXiv preprint arXiv:2409.00588},
    year={2024}
}
"""

import copy
import torch
import torch.nn as nn
import logging
from torch.distributions import Normal
from collections import namedtuple

log = logging.getLogger(__name__)

Sample = namedtuple("Sample", "trajectories chains")

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def make_timesteps(batch_size, t, device):
    return torch.full((batch_size,), t, device=device, dtype=torch.long)


class LeRobotVPGDiffusion(nn.Module):
    """
    VPG Wrapper specifically rewritten to natively wrap a Hugging Face LeRobot DiffusionPolicy.
    Handles 'proprio' keys, LeRobot normalization, and DPPO logprob math.
    """
    def __init__(
        self,
        actor,               # Hydra will pass your instantiated LeRobot DiffusionPolicy here
        critic,
        ft_denoising_steps,
        horizon_steps,
        action_dim,
        device="cuda:0",
        randn_clip_value=10,
        final_action_clip_value=None,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        
        # --- 1. LEROBOT ACTOR SETUP ---
        self.actor = actor.to(device)
        self.actor_ft = copy.deepcopy(self.actor)
        
        # Freeze the pre-trained base policy
        for param in self.actor.parameters():
            param.requires_grad = False
            
        # --- 2. EXTRACT DIFFUSERS SCHEDULE FROM LEROBOT ---
        # LeRobot uses diffusers.DDPMScheduler. We extract its math to compute PPO logprobs
        scheduler = self.actor.noise_scheduler
        self.denoising_steps = scheduler.config.num_train_timesteps
        self.ft_denoising_steps = ft_denoising_steps
        
        self.alphas_cumprod = scheduler.alphas_cumprod.clone().to(device)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.alphas = self.alphas_cumprod / self.alphas_cumprod_prev
        self.betas = 1.0 - self.alphas
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.ddpm_var = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        
        self.ddpm_mu_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ddpm_mu_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

        # --- 3. DPPO CLIPPING ---
        self.randn_clip_value = randn_clip_value
        self.final_action_clip_value = final_action_clip_value
        self.min_sampling_denoising_std = min_sampling_denoising_std
        self.min_logprob_denoising_std = min_logprob_denoising_std
        
        # --- 4. CRITIC ---
        self.critic = critic.to(device)

    def get_min_sampling_denoising_std(self):
        return self.min_sampling_denoising_std

    def _predict_noise(self, policy, x, t, cond):
        """Routes data securely through LeRobot's internal encoders and UNet."""

        batch = {}
        if "proprio" in cond:
            batch["observation.state"] = cond["proprio"]
        if "rgb" in cond:
            batch["observation.images.rgb"] = cond["rgb"]
            
        norm_batch = policy.normalize_inputs(batch)
        global_cond = policy.model.get_global_cond(norm_batch) 
        
        noise = policy.model.unet(sample=x, timestep=t, global_cond=global_cond)
        return noise

    def p_mean_var(self, x, t, cond, use_base_policy=False, deterministic=False):
        policy = self.actor if use_base_policy else self.actor_ft
        
        noise = self._predict_noise(policy, x, t, cond)
        
        x_recon = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
        )
        
        mu = (
            extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
            + extract(self.ddpm_mu_coef2, t, x.shape) * x
        )
        logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
        etas = torch.ones_like(mu).to(mu.device)
        return mu, logvar, etas

    @torch.no_grad()
    def forward(self, cond, deterministic=False, return_chain=True, use_base_policy=False):
        """The inference loop. Generates trajectories for the MuJoCo rollout."""
        sample_data = cond.get("proprio", cond.get("state", cond.get("rgb")))
        B = len(sample_data)
        
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=self.device)
        t_all = list(reversed(range(self.denoising_steps)))
        
        chain = [] if return_chain else None
        if self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
            
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, self.device)
            mean, logvar, _ = self.p_mean_var(
                x=x, t=t_b, cond=cond, use_base_policy=use_base_policy, deterministic=deterministic
            )
            std = torch.exp(0.5 * logvar)
            
            if deterministic and t == 0:
                std = torch.zeros_like(std)
            elif deterministic:
                std = torch.clip(std, min=1e-3)
            else:
                std = torch.clip(std, min=self.min_sampling_denoising_std)
                
            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise
            
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
                
            if return_chain and t <= self.ft_denoising_steps:
                chain.append(x)
                
        if return_chain:
            chain = torch.stack(chain, dim=1)
            
        policy = self.actor if use_base_policy else self.actor_ft
        unnorm_dict = policy.unnormalize_outputs({"action": x})
        final_action = unnorm_dict["action"]
        
        return Sample(final_action, chain)

    def get_logprobs(self, cond, chains, get_ent=False, use_base_policy=False):
        """Used for Behavior Cloning (BC) constraints against the base policy"""
        cond = {
            key: cond[key]
            .unsqueeze(1)
            .repeat(1, self.ft_denoising_steps, *(1,) * (cond[key].ndim - 1))
            .flatten(start_dim=0, end_dim=1)
            for key in cond
        }
        
        t_single = torch.arange(start=self.ft_denoising_steps - 1, end=-1, step=-1, device=self.device)
        t_all = t_single.repeat(chains.shape[0], 1).flatten()
        
        chains_prev = chains[:, :-1].reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains[:, 1:].reshape(-1, self.horizon_steps, self.action_dim)
        
        next_mean, logvar, eta = self.p_mean_var(chains_prev, t_all, cond, use_base_policy=use_base_policy)
        
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)
        
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def get_logprobs_subsample(self, cond, chains_prev, chains_next, denoising_inds, get_ent=False, use_base_policy=False):
        """Used heavily during PPO Advantage Updates"""
        t_single = torch.arange(start=self.ft_denoising_steps - 1, end=-1, step=-1, device=self.device)
        t_all = t_single[denoising_inds]
        
        next_mean, logvar, eta = self.p_mean_var(chains_prev, t_all, cond, use_base_policy=use_base_policy)
        
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)
        
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob



"""
DPPO: Diffusion Policy Policy Optimization. 

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""

from typing import Optional
import torch
import logging
import math

log = logging.getLogger(__name__)
from model.diffusion.diffusion_vpg import VPGDiffusion


class PPODiffusion(LeRobotVPGDiffusion):
    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate

        # Clipping value for value loss
        self.clip_vloss_coef = clip_vloss_coef

        # Discount factor for diffusion MDP
        self.gamma_denoising = gamma_denoising

        # Quantiles for clipping advantages
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):
        """
        PPO loss

        obs: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
            rgb: (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B, )
        advantages: (B,)
        oldlogprobs: (B, K, Ta, Da)
        use_bc_loss: whether to add BC regularization loss
        reward_horizon: action horizon that backpropagates gradient
        """
        # Get new logprobs for denoising steps from T-1 to 0 - entropy is fixed fod diffusion
        newlogprobs, eta = self.get_logprobs_subsample(
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            get_ent=True,
        )
        entropy_loss = -eta.mean()
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)

        # only backpropagate through the earlier steps (e.g., ones actually executed in the environment)
        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]

        # Get the logprobs - batch over B and denoising steps
        newlogprobs = newlogprobs.mean(dim=(-1, -2)).view(-1)
        oldlogprobs = oldlogprobs.mean(dim=(-1, -2)).view(-1)

        bc_loss = 0
        if use_bc_loss:
            # See Eqn. 2 of https://arxiv.org/pdf/2403.03949.pdf
            # Give a reward for maximizing probability of teacher policy's action with current policy.
            # Actions are chosen along trajectory induced by current policy.

            # Get counterfactual teacher actions
            samples = self.forward(
                cond=obs,
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )
            # Get logprobs of teacher actions under this policy
            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,
                get_ent=False,
                use_base_policy=False,
            )
            bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
            bc_logprobs = bc_logprobs.mean(dim=(-1, -2)).view(-1)
            bc_loss = -bc_logprobs.mean()

        # normalize advantages
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clip advantages by 5th and 95th percentile
        advantage_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        advantage_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=advantage_min, max=advantage_max)

        # denoising discount
        discount = torch.tensor(
            [
                self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
                for i in denoising_inds
            ]
        ).to(self.device)
        advantages *= discount

        # get ratio
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # exponentially interpolate between the base and the current clipping value over denoising steps and repeat
        t = (denoising_inds.float() / (self.ft_denoising_steps - 1)).to(self.device)
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = t

        # get kl difference and whether value clipped
        with torch.no_grad():
            # old_approx_kl: the approximate Kullback–Leibler divergence, measured by (-logratio).mean(), which corresponds to the k1 estimator in John Schulman’s blog post on approximating KL http://joschu.net/blog/kl-approx.html
            # approx_kl: better alternative to old_approx_kl measured by (logratio.exp() - 1) - logratio, which corresponds to the k3 estimator in approximating KL http://joschu.net/blog/kl-approx.html
            # old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_ploss_coef).float().mean().item()

        # Policy loss with clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss optionally with clipping
        newvalues = self.critic(obs).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
            eta.mean().item(),
        )
