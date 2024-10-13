# AMB: Adversarial MARL Benchmark

[Documents](https://marl.aisafety.org.cn/doc/index.html) [中文文档](https://marl.aisafety.org.cn/doc/index.html)

## 简介

“逐鹿”多智能体强化学习鲁棒性评测平台是由北京航空航天大学复杂关键软件环境全国重点实验室刘祥龙教授实验室开发的，对于多智能体强化学习的鲁棒性进行评测的平台。

平台实现评测标准多样化、鲁棒性能可调优、评测结果可落地、模型环境可自定的多智能体强化学习鲁棒性一体化评测流程。

对于评测标准多样化而言，本平台根据马尔可夫决策过程中的状态、动作、环境、奖励函数四个维度，模拟真实环境中的多样化不确定性，对于多智能体强化学习训练、部署过程中的多样化鲁棒性缺失问题进行全面、一体化的评测，集成超过10种鲁棒性评测方法，保障算法落地过程中全面、可靠的部署。

对于鲁棒性能可调优而言，本平台集成超过20种训练技巧，并支持对于不同算法、不同环境子任务、不同鲁棒性种类的训练技巧评测。用户可以根据自身部署场景的需要，自行根据需要的鲁棒性种类，选取本平台中最优的算法及训练技巧，或对于自身设计的算法进行参数调优，从而保障算法部署过程中的最优性能。

对于评测结果可落地而言，本平台集成无人机控制、无人车控制、电网控制、灵巧手控制等高保真多智能体强化学习环境，并对于这些环境进行全方位的鲁棒性评价与测试，其结果将有助于这些领域的工作者在应用多智能体强化学习算法的过程中，评估不同多智能体强化学习算法在其领域内的性能和鲁棒性，并决定使用哪种多智能体强化学习算法、哪种多智能体强化学习技巧进行训练。

对于模型环境可自定而言，本平台将多智能体强化学习智能体抽象为Agent类，对于本平台未集成的自定义多智能体强化学习算法实现，只需令待评测智能体满足Agent类规定的接口，即可将其上传至本平台进行评测。对于自定义环境而言，本平台满足任意基于Gym环境接口的算法评测，用户可以根据自身需要，灵活方便地进行自定义。

## 项目成员

刘祥龙 教授； 李思民 博士； 于鑫 博士； 景宗雷 博士； 郑宇威 硕士； 徐睿霄 硕士；


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

### Install network system environment dependencies

```shell
pip install sumo sumolib traci
```

### Install voltage control environment dependencies

```shell
pip install pandapower
```

## Usage Examples

### Single Algorithm Training

```bash
python -u single_train.py --env <env_name> --algo <algo_name> --exp_name <exp_name> --run single
```

### Perturbation-based Attack

```bash
python -u single_train.py --env <env_name> --algo <perturbation_algo_name> --exp_name <exp_name> --run perturbation --victim <victim_algo_name> --load_victim <dir/to/your/logdir>
```

### Adversarial Traitors

```bash
python -u single_train.py --env <env_name> --algo <traitor_algo_name> --exp_name <exp_name> --run traitor --victim <victim_algo_name> --load_victim <dir/to/your/logdir>
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
