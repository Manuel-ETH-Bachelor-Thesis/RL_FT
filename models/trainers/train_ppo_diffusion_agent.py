"""
Class for fine-tuning pre-trained diffusion model using Diffusion PPO.
Adapted from
@inproceedings{dppo2024,
    title={Diffusion Policy Policy Optimization},
    author={Ren, Allen Z. and Lidard, Justin and Ankile, Lars L. and Simeonov, Anthony and Agrawal, Pulkit and Majumdar, Anirudha and Burchfiel, Benjamin and Dai, Hongkai and Simchowitz, Max},
    booktitle={arXiv preprint arXiv:2409.00588},
    year={2024}
}
"""

import os
import math
import pickle
import random
import logging
from typing import Optional

import numpy as np
import torch
import hydra
import wandb
import einops
from omegaconf import OmegaConf
from collections import deque

from models.utils.gym_utils import make_async 
from models.utils.timer import Timer
from models.utils.scheduler import CosineAnnealingWarmupRestarts
from models.utils.reward_scaling import RunningRewardScaler
from models.utils.modules import RandomShiftsAug

log = logging.getLogger(__name__)

class TrainPPODiffusionAgent:
    def __init__(self, cfg):
        self.cfg = cfg  
        self.device = cfg.model.device
        self.seed = cfg.model.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # ==========================================
        # 1. Base Agent Setup
        # ==========================================
        
        self.use_wandb = cfg.model.get("wandb", None) is not None  
        if self.use_wandb:  
            wandb.init(
                entity=cfg.model.wandb.entity,  
                project=cfg.model.wandb.project,  
                name=cfg.model.wandb.run,  
                config=OmegaConf.to_container(cfg, resolve=True),  
            )

        self.env_name = cfg.env.name  
        self.n_envs = cfg.env.n_envs  
        self.n_cond_step = cfg.env.cond_steps  
        self.obs_dim = cfg.env.obs_dim  
        self.action_dim = cfg.env.action_dim  
        self.act_steps = cfg.env.act_steps  
        self.horizon_steps = cfg.env.horizon_steps  
        self.max_episode_steps = cfg.env.max_episode_steps  
        
        # Must instantiate self.venv from the main execution script!
        self.venv = None 

        self.batch_size = cfg.model.train.batch_size  
        
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        if "base_policy_path" not in cfg:
            raise ValueError("LeRobot requires a base_policy_path directory!")
        log.info(f"Loading pre-trained LeRobot policy from {cfg.base_policy_path}")
        lerobot_actor = DiffusionPolicy.from_pretrained(cfg.base_policy_path)
        self.model = hydra.utils.instantiate(cfg.model, actor=lerobot_actor).to(self.device)

        self.itr = 0
        self.n_train_itr = cfg.model.train.n_train_itr  
        self.val_freq = cfg.model.train.val_freq  
        self.force_train = cfg.model.train.get("force_train", False)  
        self.n_steps = cfg.model.train.n_steps  
        self.max_grad_norm = cfg.model.train.get("max_grad_norm", None)  

        self.logdir = cfg.model.logdir  
        self.render_dir = os.path.join(self.logdir, "render")
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        self.result_path = os.path.join(self.logdir, "result.pkl")
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.save_trajs = cfg.model.train.get("save_trajs", False)  
        self.log_freq = cfg.model.train.get("log_freq", 1)  
        self.save_model_freq = cfg.model.train.save_model_freq  
        self.render_freq = cfg.model.train.render.freq  
        self.n_render = cfg.model.train.render.num  
        self.render_video = cfg.env.get("save_video", False)  
        
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (self.n_render <= 0 and self.render_video), "Need to set n_render > 0 if saving video"

        # ==========================================
        # 2. PPO Setup (Mapped to cfg.model.train.*)
        # ==========================================
        
        self.logprob_batch_size = cfg.model.train.get("logprob_batch_size", 10000)  
        assert self.logprob_batch_size % self.n_envs == 0, "logprob_batch_size must be divisible by n_envs"

        self.gamma = cfg.model.train.gamma  
        self.n_critic_warmup_itr = cfg.model.train.n_critic_warmup_itr  

        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=cfg.model.train.actor_lr,  
            weight_decay=cfg.model.train.actor_weight_decay,  
        )
        
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.model.train.actor_lr_scheduler.first_cycle_steps,  
            cycle_mult=1.0,
            max_lr=cfg.model.train.actor_lr,  
            min_lr=cfg.model.train.actor_lr_scheduler.min_lr,  
            warmup_steps=cfg.model.train.actor_lr_scheduler.warmup_steps,  
            gamma=1.0,
        )
        
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=cfg.model.train.critic_lr,  
            weight_decay=cfg.model.train.critic_weight_decay,  
        )
        
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg.model.train.critic_lr_scheduler.first_cycle_steps,  
            cycle_mult=1.0,
            max_lr=cfg.model.train.critic_lr,  
            min_lr=cfg.model.train.critic_lr_scheduler.min_lr,  
            warmup_steps=cfg.model.train.critic_lr_scheduler.warmup_steps,  
            gamma=1.0,
        )

        self.gae_lambda = cfg.model.train.get("gae_lambda", 0.95)  
        self.target_kl = cfg.model.train.get("target_kl", None)  
        self.update_epochs = cfg.model.train.update_epochs  
        self.ent_coef = cfg.model.train.get("ent_coef", 0)  
        self.vf_coef = cfg.model.train.get("vf_coef", 0)  

        self.reward_scale_running = cfg.model.train.get("reward_scale_running", False)  
        if self.reward_scale_running:
            self.running_reward_scaler = RunningRewardScaler(self.n_envs)

        self.reward_scale_const = cfg.model.train.get("reward_scale_const", 1.0)  
        self.use_bc_loss = cfg.model.train.get("use_bc_loss", False)  
        self.bc_loss_coeff = cfg.model.train.get("bc_loss_coeff", 0)  

        # ==========================================
        # 3. Diffusion Setup (Mapped to cfg.model.train.*)
        # ==========================================
        
        self.reward_horizon = cfg.env.get("reward_horizon", self.act_steps)  

        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.model.train.eta_update_interval  
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.model.train.eta_lr,  
                weight_decay=cfg.model.train.eta_weight_decay,  
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.model.train.eta_lr_scheduler.first_cycle_steps,  
                cycle_mult=1.0,
                max_lr=cfg.model.train.eta_lr,  
                min_lr=cfg.model.train.eta_lr_scheduler.min_lr,  
                warmup_steps=cfg.model.train.eta_lr_scheduler.warmup_steps,  
                gamma=1.0,
            )
            
        self.grad_accumulate = cfg.model.train.get("grad_accumulate", 1)
        self.augment = cfg.model.train.get("augment", True)
        if self.augment:
            self.aug = RandomShiftsAug(pad=4) 

    def save_model(self):
        data = {
            "itr": self.itr,
            "model": self.model.state_dict(),
        }  
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")

    def load(self, itr):
        loadpath = os.path.join(self.checkpoint_dir, f"state_{itr}.pt")
        data = torch.load(loadpath, weights_only=True)
        self.itr = data["itr"]
        self.model.load_state_dict(data["model"])

    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        
        assert self.venv is not None, "Execution script must instantiate agent.venv = env before calling agent.run()"
        
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(f"<-- Reset environment {index} with options {options_venv[index]}")
        return obs_venv

    def run(self):
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        
        assert self.venv is not None, "Execution script must set self.venv prior to running."
        
        while self.itr < self.n_train_itr:
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.cfg.env.get("reset_at_iteration", True) or eval_mode or last_itr_eval:
                prev_obs_raw = self.reset_env_all(options_venv=options_venv)

                obs_history = {
                    "proprio": deque(maxlen=self.n_cond_step),
                    "rgb": deque(maxlen=self.n_cond_step),
                    "depth": deque(maxlen=self.n_cond_step),
                }

                for _ in range(self.n_cond_step):
                    obs_history["proprio"].append(prev_obs_raw["proprio"])
                    obs_history["rgb"].append(prev_obs_raw["rgb"])
                    obs_history["depth"].append(prev_obs_raw["depth"])
                firsts_trajs[0] = 1
            else:
                firsts_trajs[0] = done_venv

            obs_trajs = {
                "proprio": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, 62), dtype=np.float32),
                "rgb": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, 3, 128, 128), dtype=np.uint8),
                "depth": np.zeros((self.n_steps, self.n_envs, self.n_cond_step, 1, 128, 128), dtype=np.uint8)
            } 

            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            truncated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                with torch.no_grad():
                    stacked_obs = {
                        "proprio": np.stack(obs_history["proprio"], axis=1),
                        "rgb": np.stack(obs_history["rgb"], axis=1),
                        "depth": np.stack(obs_history["depth"], axis=1),
                    }
                    cond = {
                        "proprio": torch.from_numpy(stacked_obs["proprio"]).float().to(self.device),
                        "rgb": torch.from_numpy(stacked_obs["rgb"]).float().to(self.device) / 255.0,
                        "depth": torch.from_numpy(stacked_obs["depth"]).float().to(self.device) / 255.0,
                    }
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=not eval_mode,
                    )
                    output_venv = samples.trajectories.cpu().numpy() 
                    chains_venv = samples.chains.cpu().numpy() if samples.chains is not None else 0                    
                
                action_venv = output_venv[:, : self.act_steps]

                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv
                    
                obs_trajs["proprio"][step] = stacked_obs["proprio"]
                obs_trajs["rgb"][step] = stacked_obs["rgb"]
                obs_trajs["depth"][step] = stacked_obs["depth"]
                
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                truncated_trajs[step] = truncated_venv
                firsts_trajs[step + 1] = done_venv
                obs_history["proprio"].append(obs_venv["proprio"])
                obs_history["rgb"].append(obs_venv["rgb"])
                obs_history["depth"].append(obs_venv["depth"])
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
                        
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array([np.sum(reward_traj) for reward_traj in reward_trajs_split]) 
                avg_episode_reward = np.mean(episode_reward)
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                log.info("[WARNING] No episode completed within the iteration!")

            if not eval_mode:
                with torch.no_grad():
                    obs_trajs["proprio"] = torch.from_numpy(obs_trajs["proprio"]).float().to(self.device)
                    obs_trajs["rgb"] = torch.from_numpy(obs_trajs["rgb"]).float().to(self.device) / 255.0
                    obs_trajs["depth"] = torch.from_numpy(obs_trajs["depth"]).float().to(self.device) / 255.0

                    if self.augment:
                        rgb = einops.rearrange(obs_trajs["rgb"], "s e t c h w -> (s e) (t c) h w")
                        rgb = self.aug(rgb)
                        obs_trajs["rgb"] = einops.rearrange(rgb, "(s e) (t c) h w -> s e t c h w", s=self.n_steps, e=self.n_envs, t=self.n_cond_step)
                        
                        depth = einops.rearrange(obs_trajs["depth"], "s e t c h w -> (s e) (t c) h w")
                        depth = self.aug(depth)
                        obs_trajs["depth"] = einops.rearrange(depth, "(s e) (t c) h w -> s e t c h w", s=self.n_steps, e=self.n_envs, t=self.n_cond_step)
                    
                    obs_k = {
                        "proprio": einops.rearrange(obs_trajs["proprio"], "s e ... -> (s e) ..."),
                        "rgb": einops.rearrange(obs_trajs["rgb"], "s e ... -> (s e) ..."),
                        "depth": einops.rearrange(obs_trajs["depth"], "s e ... -> (s e) ..."),
                    }
                    num_split = math.ceil((self.n_envs * self.n_steps) / self.logprob_batch_size)
                    obs_ts = []
                    for i in range(num_split):
                        start = i * self.logprob_batch_size
                        end = start + self.logprob_batch_size
                        
                        obs_chunk = {
                            "proprio": obs_k["proprio"][start:end],
                            "rgb": obs_k["rgb"][start:end],
                            "depth": obs_k["depth"][start:end],
                        }
                        obs_ts.append(obs_chunk)
                        
                    values_list = []
                    for obs in obs_ts:
                        values = self.model.critic(obs).cpu().numpy().flatten()
                        values_list.append(values.reshape(-1, self.n_envs))
                    values_trajs = np.concatenate(values_list, axis=0)
                    values_trajs = np.asarray(values_trajs).reshape(self.n_steps, self.n_envs)
                    chains_t = einops.rearrange(torch.from_numpy(chains_trajs).float().to(self.device), "s e t h d -> (s e) t h d")
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    
                    logprobs_trajs = np.empty((0, self.model.ft_denoising_steps, self.horizon_steps, self.action_dim))
                    
                    logprobs_list = []
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_list.append(logprobs)
                    logprobs_trajs = np.concatenate(logprobs_list, axis=0)
                    
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(reward=reward_trajs.T, first=firsts_trajs[:-1].T)
                        reward_trajs = reward_trajs_transpose.T

                    obs_venv_ts = {
                        "proprio": torch.from_numpy(obs_venv["proprio"]).float().to(self.device),
                        "rgb": torch.from_numpy(obs_venv["rgb"]).float().to(self.device) / 255.0,
                        "depth": torch.from_numpy(obs_venv["depth"]).float().to(self.device) / 255.0
                    }
                    
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = self.model.critic(obs_venv_ts).detach().cpu().numpy().reshape(-1)
                        else:
                            nextvalues = values_trajs[t + 1]
                        nextvalues = nextvalues.squeeze()
                        
                        # Fix: use terminated_trajs to ensure proper value bootstrapping on truncation
                        nonterminal = 1.0 - terminated_trajs[t]
                        delta = (reward_trajs[t] * self.reward_scale_const + self.gamma * nextvalues * nonterminal - values_trajs[t])
                        advantages_trajs[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                    returns_trajs = advantages_trajs + values_trajs

                chains_k = einops.rearrange(torch.from_numpy(chains_trajs).to(self.device).float(), "s e t h d -> (s e) t h d")
                returns_k = torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                values_k = torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                advantages_k = torch.tensor(advantages_trajs, device=self.device).float().reshape(-1)
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                clipfracs = []

                for update_epoch in range(self.update_epochs):
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    if self.learn_eta:
                        self.eta_optimizer.zero_grad()

                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end] 
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b, (self.n_steps * self.n_envs, self.model.ft_denoising_steps)
                        )
                        obs_b = {
                            "proprio": obs_k["proprio"][batch_inds_b],
                            "rgb": obs_k["rgb"][batch_inds_b],
                            "depth": obs_k["depth"][batch_inds_b],
                        }
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        (
                            pg_loss, entropy_loss, v_loss, clipfrac, approx_kl, ratio, bc_loss, eta,
                        ) = self.model.loss(
                            obs_b, chains_prev_b, chains_next_b, denoising_inds_b,
                            returns_b, values_b, advantages_b, logprobs_b,
                            use_bc_loss=self.use_bc_loss, reward_horizon=self.reward_horizon,
                        )
                        loss = pg_loss + entropy_loss * self.ent_coef + v_loss * self.vf_coef + bc_loss * self.bc_loss_coeff
                        clipfracs += [clipfrac]

                        (loss / self.grad_accumulate).backward()

                        if (batch + 1) % self.grad_accumulate == 0 or batch == num_batch - 1:
                            if self.itr >= self.n_critic_warmup_itr:
                                if self.max_grad_norm is not None:
                                    torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
                                self.actor_optimizer.step()
                                if self.learn_eta and (batch // self.grad_accumulate) % self.eta_update_interval == 0:
                                    self.eta_optimizer.step()
                                    
                            self.critic_optimizer.step()
                            
                            self.actor_optimizer.zero_grad()
                            self.critic_optimizer.zero_grad()
                            if self.learn_eta:
                                self.eta_optimizer.zero_grad()

                        approx_kl_val = approx_kl.item() if isinstance(approx_kl, torch.Tensor) else approx_kl
                        log.info(f"approx_kl: {approx_kl_val}, update_epoch: {update_epoch}, num_batch: {num_batch}")
                        
                        if self.target_kl is not None and approx_kl_val > self.target_kl:
                            flag_break = True
                            break
                            
                    if flag_break:
                        break

                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            run_results.append({"itr": self.itr, "step": cnt_train_step})
                
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(f"eval: avg episode reward {avg_episode_reward:8.4f}")
                    if self.use_wandb:
                        wandb.log({
                            "avg episode reward - eval": avg_episode_reward,
                            "num episode - eval": num_episode_finished,
                        }, step=self.itr, commit=False)
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                else:
                    log.info(f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}")
                    if self.use_wandb:
                        wandb.log({
                            "total env step": cnt_train_step,
                            "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
                            "pg loss": pg_loss.item() if isinstance(pg_loss, torch.Tensor) else pg_loss,
                            "value loss": v_loss.item() if isinstance(v_loss, torch.Tensor) else v_loss,
                            "bc loss": bc_loss.item() if isinstance(bc_loss, torch.Tensor) else bc_loss,
                            "eta": eta,
                            "approx kl": approx_kl_val,
                            "ratio": ratio.item() if isinstance(ratio, torch.Tensor) else ratio,
                            "clipfrac": np.mean(clipfracs),
                            "explained variance": explained_var,
                            "avg episode reward - train": avg_episode_reward,
                            "num episode - train": num_episode_finished,
                            "diffusion - min sampling std": diffusion_min_sampling_std,
                            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                            "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                        }, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1