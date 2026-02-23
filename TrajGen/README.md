# Trajectory generation for DreamControl

## Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.

Setup conda env:
```shell
conda env create -f environment.yml
conda activate omnicontrol
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Install JAX and JAXLIB with cuda support (Replace cuda12 with your cuda version):
```bash
conda activate dreamcontrol # As latest JAX requires python 3.10, so we need to use the dreamcontrol environment to install JAX.
pip install jax[cuda12]
```

### Download the pretrained model

Download the following model and place it in `./save/`. 

[model_humanml](https://drive.google.com/file/d/1oTkBtArc3xjqkYD6Id7LksrTOn3e1Zud/view?usp=sharing)


## Task-specific Motion Synthesis

To run trajectory generation for a task, run the following command:

```bash
bash collect_<TASK_NAME>.sh
```

TASK_NAME can be one of the following:
```
- Pick
- Button_Press
- Punch
- Kick
- Bimanual_Pick
- Ground_Pick
- Ground_Pick_Top
- Open_Drawer
- Open_Door
- Jump
- Sit
- Pick_Place
```


For instance, to run trajectory generation for ```Button_Press``` task, run the following command:

```bash
bash collect_Button_Press.sh
```

This will generate the trajectories in the `sample/<TASK_NAME>_sim2` folder. This will first generate human reference trajectories in `sample/<TASK_NAME>_sim` folder, and then generate retargeted trajectories in `sample/<TASK_NAME>_sim1` folder. Later, it will refine the retargeted trajectories and save them in `sample/<TASK_NAME>_sim2` folder. For task-specific retargeting and refinement, refer to `sample/<TASK_NAME>_sim/retarget.py` and `sample/<TASK_NAME>_sim1/refine_motions.py` scripts. To visualize the trajectories, run the following command:

```bash
cd sample
python visualize_trajectories.py --folder <FOLDER_NAME> --id <ID>
```

For instance, to visualize the trajectory for ```Button_Press``` task with trajectory ID ```0_n```, run the following command:

```bash
python visualize_trajectories.py --folder Button_Press_sim2 --id 0_n
```

## Code pointer to the main module of OmniControl
[Spatial Guidance](./diffusion/gaussian_diffusion.py#L450). (./diffusion/gaussian_diffusion.py#L450)  
[Realism Guidance](./model/cmdm.py#L158). (./model/cmdm.py#L158)

## Acknowledgments

This part of the code is based on [OmniControl](https://github.com/neu-vi/omnicontrol). We thank the authors for their contributions.

## License
Note that this part of the code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
