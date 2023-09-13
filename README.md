# Adversarial MARL Benchmark

## Requirements

* Python >= 3.8
* PyTorch >= 1.12.0

## Support Features

### Pipelines

* Single MARL training
* Perturbation-based attacks & traitors
* Adversarial traitors
* Dual MARL training

### Algorithms

* MAPPO
* MADDPG

### Environments

* SMAC
* SMACv2
* Multi-Agent MuJoCo
* PettingZoo MPE
* Google Research Football
* Gym
* Toy Example
* Custom environments

## Install Dependencies

### Install Python requirements

```bash
pip install -r requirements.txt
```

### Install SMAC

Change to the directory where you want to install StarCraftII, then run following commands:

```bash
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip 
rm -rf SC2.4.10.zip

cd StarCraftII/
wget https://raw.githubusercontent.com/Blizzard/s2client-proto/master/stableid.json

pip install pysc2
# enum34 is deprecated in newer python version, and should be deleted.
pip uninstall enum34
```

Add following lines into `~/.bashrc`:

```bash
export SC2PATH="/path/to/your/StarCraftII"
```

Copy the `amb/envs/smac/SMAC_Maps` directory to `StarCraftII/Maps`.

### Install Multi-Agent MuJoCo

First, install MuJoCo by running the following commands:

```bash
mkdir ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar xzvf mujoco210-linux-x86_64.tar.gz
rm -rf mujoco210-linux-x86_64.tar.gz
```


Add following commands to your `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/your/.mujoco/mujoco210/bin
```

Then, install mujoco-py by running the following commands:

```bash
# ubuntu only. If there are more troubles (e.g., -lGL cannot be found, etc.), please refer to the official github of mujoco-py.
sudo apt install libglew-dev libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
# newer Cython version will cause error
pip install "Cython<3"
# install mujoco-py
pip install "mujoco-py<2.2,>=2.1"
```

Running following commands in Python to make sure mujoco is installed.

```python
import mujoco_py
import os
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

print(sim.data.qpos)
```

### Install Pettingzoo MPE

```bash
pip install pettingzoo==1.22.0 supersuit==3.7.0
```

### Install Google Research Footabll

```bash
pip install gfootball
```

### Solving dependencies

Run following commands after all enviroments are installed.

```bash
pip install gym==0.21.0 pyglet==1.5.0 importlib-metadata==4.13.0 numpy==1.21.5 protobuf==3.20.3
```

## Usage Examples

### Single Algorithm Training

```bash
python -u single_train.py --env <env_name> --algo <algo_name> --exp_name <exp_name> --run single
```

### Perturbation-based Attack

```bash
python -u single_train.py --env <env_name> --algo <perturbation_algo_name> --exp_name <exp_name> --run perturbation --victim <victim_algo_name> --victim.model_dir <dir/to/your/model>
```

### Adversarial Traitors

```bash
python -u single_train.py --env <env_name> --algo <traitor_algo_name> --exp_name <exp_name> --run traitor --victim <victim_algo_name> --victim.model_dir <dir/to/your/model>
```

### Dual Algorithm Training

```bash
# In dual training, "angel" and "demon" are two competitive teams, where we only train "angel" but fix "demon".
python -u dual_train.py --env <env_name> --angel <angel_algo_name> --demon <demon_algo_name> --exp_name <exp_name> --run dual
```

### Load Victim Config from Directory

```bash
# It will load environment and victim configurations from JSON, together with the victim's checkpoints in "models/" directory
python -u single_train.py --algo <adv_algo_name> --exp_name <exp_name> --run [traitor|perturbation] --load_victim <dir/to/victim/results>
# In dual training, you can load angel and demon separately, even from single training checkpoint.
python -u dual_train.py --env <env_name> --load_angel <dir/to/angel/results> --load_victim <dir/to/demon/results> --exp_name <exp_name> --run dual
```

### Argument Decided Parameters

```bash
# Here is an example. "A.B" -> A means namespace (algo, env, victim), and B means the parameter name.
python -u single_train.py --algo.lr 5e-4 --victim.share_param False --env.map_name 2s3z
```