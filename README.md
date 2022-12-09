# Legged Nav

This is the official implementation of **Rethinking Sim2Real: Lower Fidelity Simulation Leads to Higher Sim2Real Transfer in Navigation**. Use this repo to train legged robots to navigate kinematically or dynamically. We currently support A1, AlienGo, and Spot. 

## Setup

This project is builds off of `habitat-lab` and `habitat-sim`

1. Initialize the project
```bash
mkdir legged_nav
cd legged_nav 
git clone --branch kin2dyn git@github.com:joannetruong/habitat-lab.git
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim && git checkout 1fb3f693e40279db09d0e0c9e5fa1357c30ab03c
```
2. Create a conda environment
```bash
conda create -n legged_nav -y python=3.7 cmake=3.14.0 
conda activate legged_nav
```
3. Install Habitat-Sim
```bash
pip install -r requirements.txt
python setup.py install --bullet --headless
```
4. Install Habitat-Lab
```bash
cd ../habitat-lab
pip install typing-extensions~=3.7.4 google-auth==1.6.3 simplejson braceexpand pybullet cython pkgconfig squaternion
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
python setup.py develop --all
```

## Data
1. Download scene datasets for HM3D and Gibson following [instructions here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md). Extract the scenes to `data/scene_datasets/`
2. Download training and validation episodes

| Robot | Download Episodes | Extract Path |
|-|-|-|
| A1 | [Link](https://drive.google.com/file/d/1mbjHy09KIKyFb4atX_j38WT-4MQvO3aW/view?usp=share_link) | `data/pointnav_hm3d_gibson/pointnav_a1_0.2` |
| AlienGo | [Link](https://drive.google.com/file/d/1q05VcaqMzvWPaq_sXqudWTJsUSnPjt2a/view?usp=share_link) | `data/pointnav_hm3d_gibson/pointnav_aliengo_0.22` |
| Spot | [Link](https://drive.google.com/file/d/14vKI-AZmmxS5cQg1lV0ybFsuRNg4VMeL/view?usp=share_link) | `data/pointnav_hm3d_gibson/pointnav_spot_0.3` |
3. Download robot URDFs

| Robot | Download URDFs | Extract Path |
|-|-|-|
| A1 | [Link](https://drive.google.com/file/d/1xpqcpBaFf1ld415mYOHfDCoA-oMdLGVr/view?usp=share_link) | `data/URDF_demo_assets/a1` |
| AlienGo | [Link](https://drive.google.com/file/d/1PuS0pJmFqBD-BuxvOQRVTgWScq5vWH06/view?usp=share_link) | `data/URDF_demo_assets/aliengo` |
| Spot | [Link](https://drive.google.com/file/d/1uLiR5JcFEaQ1xNAezoSdv48Zj6QORVZY/view?usp=share_link) | `data/URDF_demo_assets/spot` |
4. Please double check that your directory structure is as follows:
```graphql
data
|─ datasets
|   ├─ pointnav_hm3d_gibson
|   |    ├─ pointnav_a1_0.2
|   |    ├─ pointnav_aliengo_0.22
|   |    ├─ pointnav_spot_0.3
├─ scene_datasets
|   ├─ hm3d
|   ├─ gibson
├─ URDF_demo_assets
|   ├─ a1
|   ├─ aliengo
|   ├─ spot
```

## Run

The `run.py` script controls training and evaluation for all models and datasets:
* Modify the [experiment yaml](https://github.com/joannetruong/habitat-lab/blob/kin2dyn/habitat_baselines/config/pointnav/ddppo_pointnav_quadruped.yaml) for your specific robot
* Modify the [task yaml](https://github.com/joannetruong/habitat-lab/blob/kin2dyn/configs/tasks/pointnav_quadruped.yaml) for your specific robot. 
    * To train the robot dynamically, set [POSSIBLE_ACTIONS: ["DYNAMIC_VELOCITY_CONTROL"]](https://github.com/joannetruong/habitat-lab/blob/733eb78dfc5a660a1994b14ca52b0e9852bd717b/configs/tasks/pointnav_quadruped.yaml#L41)
```bash
python -u -m habitat_baselines.run \
     --exp-config {experiment_yaml_path} \
     --run-type {train | eval}
```

For an example, to train Spot, use:
```bash
python -u -m habitat_baselines.run \
     --exp-config habitat_baselines/config/pointnav/ddppo_pointnav_quadruped.yaml \
     --run-type train
```

If you use this repo in your research, please cite the following [paper](https://arxiv.org/abs/2207.10821):

```tex
@inproceedings{truong2022kin2dyn,
    title={Rethinking Sim2Real: Lower Fidelity Simulation Leads to Higher Sim2Real Transfer in Navigation}, 
    author={Joanne Truong and Max Rudolph and Naoki Yokoyama and Sonia Chernova and Dhruv Batra and Akshara Rai}, 
    booktitle={Conference on Robot Learning (CoRL)},
    year={2022}
}
```