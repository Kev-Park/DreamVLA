# DreamControl: Human-Inspired Whole-Body Humanoid Control for Scene Interaction via Guided Diffusion

<div align="center">

<a href="https://genrobo.github.io/DreamControl/">
  <img alt="Website" src="https://img.shields.io/badge/Website-Visit-blue?style=flat&logo=google-chrome"/>
</a>

<a href="https://www.youtube.com/watch?v=mEA3v8XLog4">
  <img alt="Video" src="https://img.shields.io/badge/Video-YouTube-red?style=flat&logo=youtube"/>
</a>

<a href="https://arxiv.org/abs/2509.14353">
  <img alt="Arxiv" src="https://img.shields.io/badge/Paper-Arxiv-b31b1b?style=flat&logo=arxiv"/>
</a>

<a href="https://www.generalrobotics.company/post/dreamcontrol-building-humanoid-ai-skills">
    <img alt="Blog" src="https://img.shields.io/badge/Blog-MyBlog-blue?style=flat&logo=wordpress"/>
</a>

<a href="https://github.com/GenRobo/DreamControl/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/GenRobo/DreamControl?style=social"/>
</a>


<img src="cover.gif" width="600px"/>

</div>

This repository contains the code (reference trajectories, training, sim2real) for the paper "DreamControl: Human-Inspired Whole-Body Humanoid Control for Scene Interaction via Guided Diffusion".

DreamControl is a ***scalable system*** for learning human-inspired whole-body humanoid control for scene interaction via guided diffusion. It uses a generative model trained on offline human motion data to generate human motion trajectories for performing varied tasks. These trajectories are retargeted to a Unitree G1 robot, and a closed-loop, task-specific RL policy is trained that can be deployed both in simulation and real-world tasks.

## TODO
- [x] Release DreamControl training code with generated reference trajectories 
- [x] Release reference trajectory generation code
- [x] Release Sim2Real code

## Table of Contents

1. [Setup](#setup)
2. [Phase 1: Generate reference trajectories](#phase-1-generate-reference-trajectories)
3. [Phase 2: RL training](#phase-2-rl-training)
4. [Phase 3: Sim2Real](#phase-3-sim2real)
5. [Acknowledgement](#acknowledgement)
6. [License](#license)
7. [Citation](#citation)


## Setup

### Install IsaacSim

Create a conda environment:-
```bash
conda create -n dreamcontrol python=3.10 -y
conda activate dreamcontrol
```

Next, install a CUDA-enabled PyTorch 2.7.0 build. This step is optional for Linux, but required for Windows to ensure a CUDA-compatible version of PyTorch is installed.
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Then, install IsaacSim:-
```bash
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

### Install dependencies for DreamControl

Install dependencies:-
```bash
sudo apt install cmake build-essential
```

Clone this repository:-
```bash
git clone https://github.com/genrobo/DreamControl.git
cd DreamControl
sudo apt update
sudo apt install git-lfs
git lfs install
git lfs pull
```

Install modified isaaclab with new environments for DreamControl:-
```bash
cd Training
./isaaclab.sh --install
cd ..
```

Install other dependencies :-

```bash
pip install -r requirements.txt
```

## Phase 1: Generate reference trajectories

NOTE: All generated trajectories are in [TrajGen/sample/](https://github.com/GenRobo/DreamControl/tree/main/TrajGen/sample) folder. These trajectories are used to train the RL policy. You may skip this step if you want to directly use these generated trajectories for training.

Instructions to generate reference trajectories are in [TrajGen/README.md](https://github.com/GenRobo/DreamControl/tree/main/TrajGen/README.md) file.

## Phase 2: RL training

All training environment definitions are in [Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1](https://github.com/GenRobo/DreamControl/tree/main/Training/source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1) folder. For training RL policy for a task, run the following command:-

```bash
cd Training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Motion-Tracking-<TASK-NAME>-v0 --headless --device cuda:1
```

```TASK-NAME``` can be one of the following:-
```
- Pick
- Bimanual-Pick
- Ground-Pick
- Ground-Pick-Top
- Button-Press
- Open-Drawer
- Open-Door
- Punch
- Kick
- Jump
- Sit
- Pick-Place
```

For instance, to run RL training for ```Pick``` task, run the following command:-

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Motion-Tracking-Pick-v0 --headless --device cuda:1
```

To run inference on the trained policy, run the following command:-

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_eval.py --task=Isaac-Motion-Tracking-<TASK NAME>-v0 --headless --video --num_envs 1000
```

For instance, to run inference on the trained policy for ```Pick``` task, run the following command:-

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_eval.py --task=Isaac-Motion-Tracking-Pick-v0 --headless --video --num_envs 1000
```

This will take the latest trained checkpoint for the task in the ```Training/logs/rsl_rl/g1/``` directory and run inference on it. The video will be saved in the same log directory where the training checkpoint was taken from.

## Phase 3: Sim2Real


All Sim2Real code is in [Sim2Real](https://github.com/GenRobo/DreamControl/tree/main/Sim2Real) folder. Instructions to run tasks on hardware are in [Sim2Real/README.md](https://github.com/GenRobo/DreamControl/tree/main/Sim2Real/README.md) file.


## Acknowledgement

Parts of the code have been adapted from [IsaacLab](https://github.com/isaac-sim/IsaacLab), [OmniControl](https://github.com/neu-vi/omnicontrol), [ASAP](https://github.com/LeCAR-Lab/ASAP), [HumanoidVerse](https://github.com/LeCAR-Lab/HumanoidVerse), [pyroki](https://github.com/chungmin99/pyroki), and [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym). We thank the authors for their contributions.

## License

Copyright (c) 2025-Present, General Robotics Technology Inc. All rights reserved.

For license information, see the LICENSE file.

## Citation

This code is part of the DreamControl paper. If you find this code helpful, please cite the following paper:

```
@article{Kalaria2025DreamControlHW,
  title={DreamControl: Human-Inspired Whole-Body Humanoid Control for Scene Interaction via Guided Diffusion},
  author={Dvij Kalaria and Sudarshan S. Harithas and Pushkal Katara and Sangkyung Kwak and Sarthak Bhagat and Shankar Sastry and Srinath Sridhar and Sai H. Vemprala and Ashish Kapoor and Jonathan Chung-Kuan Huang},
  journal={ArXiv},
  year={2025},
  volume={abs/2509.14353},
}
```
