# Policy Fine-Tuning and Distillation of Pre-Trained Diffusion Models for High-Frequency Robot Control

This repository contains the official codebase for the Bachelor thesis focusing on fine-tuning diffusion policies and distilling them into a lightweight GMM-SAC actor. The environment provides an optimized, headless MuJoCo setup for a simulated pick-and-place task.

This project was built from the ground up for high data throughput, prioritizing native PyTorch/Gymnasium integration to support Reinforcement Learning and Imitation Learning pipelines.

## Prerequisites
* **Python 3.10, 3.11, or 3.12 (64-bit)**

## Installation & Setup
This repository includes a `Makefile` and `pyproject.toml` to manage dependencies. The installation command will generate a virtual environment (`.venv`), upgrade the package manager, and install dependencies and the project in editable mode.

1. Clone the repository:
   ```bash
   git clone https://github.com/Manuel-ETH-Bachelor-Thesis/RL_FT.git
   cd RL_FT
   ```
2. Build the virtual environment:
   ```bash
   make install
   ```
3. Activate the environment:
- Windows:
   ```bash
   .venv\\Scripts\\activate
   ```
- Linux / macOS:
   ```bash
   source .venv/bin/activate
   ```
To test the environment run `make test` in the activated environment and to wipe the environment run `make clean`.

## Approach
This project aims to extract relevant behaviour from pre-trained computationally heavy foundation Diffusion Models to achieve similar task performance on high-frequency edge robotics. The approach follows the Teacher-Student Distillation model:
* **The Teacher (Diffusion Policy)**: A pre-trained Diffusion Model capable of expressive, multi-modal behaviour hindered by slow inference times.
* **The Expert (DPPO)**: The Diffusion Policy is fine-tuned and generalized using Diffusion Proximal Policy Optimization in a simulated environment using reward shaping and domain randomization.
* **The Student (GMM-SAC Distillation)**: The fine-tuned behviour is distilled into an efficient Gaussian Mixture Model - Soft Actor-Critic policy enabling low latency execution on edge hardware.

## Repository
This codebase is built modularly using **Hydra** for configuration management and **dm_control** for dynamic XML scene composition. Robots, scenes, and RL algorithms can easily be swapped without rewriting Python logic.

```text
RL_FT/
├── configs/                 # Hydra configurations
│   ├── env/                 # Reward parametrization & task logic
│   ├── model/               # SAC, DPPO hyperparams
│   ├── robot/               # Robot model configs (joint names, etc.)
│   └── scene/               # Domain randomization bounds
├── envs/                    # Custom Gymnasium environments
├── models/                  # Model architecture
├── resources/               # Tailored MuJoCo Menagerie XMLs & meshes
└── scripts/                 # Training entry points & result visualizations
```

## Resources
### TODO

## Quickstart & Usage
### TODO

## Simulation & Domain Randomization
### TODO

## Results & Hardware Deployment
### TODO

## Thesis Roadmap
- [x] Implement simulation environment and evaluation suite for pick-and-place.
- [x] Establish baselines (diffusion policy only; optional public baselines if needed before the diffusion policy is ready).
- [ ] Implement diffusion model fine-tuning (policy initialisation from diffusion policy, reward shaping, domain randomisation/curriculum).
- [ ] Implement distillation of the fine-tuned diffusion model into a lightweight GMM-SAC actor.
- [ ] Run ablations (reward variants, curriculum settings) and report results.

## Acknowledgements
This Bachelor thesis is developed at **ETH Zürich** under the supervision of Prof. Dr. Christoforos Mavrogiannis, overseen by the [Computational Robotics Lab](https://crl.ethz.ch/index.html) (Prof. Dr. Stelian Coros). Upstream data gathering and diffusion policy training are enabled by the [Product Development Group Zurich](https://pdz.ethz.ch/). Hardware deployment for physical validation are facilitated in collaboration with the [Swiss Cobotics Competence Center](https://s3c.swiss/).
