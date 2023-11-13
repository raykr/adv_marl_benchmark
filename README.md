# AMB: Adversarial MARL Benchmark

## Support Features

### Pipelines

* Single MARL training
* Perturbation-based attacks & traitors
* Adversarial traitors
* Dual MARL training
* Traitors in dual MARL

### Algorithms

* MAPPO
* MADDPG
* QMIX

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

### Create Conda Environment

```bash
# This will create an Anaconda environment named amb.
conda env create -f amb.yml
```

### Install StarCraftII (for SMAC and SMACv2)

Change to the directory where you want to install StarCraftII, then run following commands:

```bash
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip 
rm -rf SC2.4.10.zip

cd StarCraftII/
wget https://raw.githubusercontent.com/Blizzard/s2client-proto/master/stableid.json
```

Add following lines into `~/.bashrc`:

```bash
export SC2PATH="/path/to/your/StarCraftII"
```

Copy the `amb/envs/smac/SMAC_Maps` and `amb/envs/smacv2/SMAC_Maps` directory to `StarCraftII/Maps`.

### Install Google Research Football

Install following dependencies (Linux only):

```shell
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip
```

Install GRF through pip:

```shell
pip install gfootball
```

### Install Bi-Dexhands

Firstly install IsaacGym correctly, Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym).

Then you maybe need to fix bugs of IsaacGym, please refer to [this issue](https://forums.developer.nvidia.com/t/attributeerror-module-numpy-has-no-attribute-float/270702)

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