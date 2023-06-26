# Adversarial MARL Benchmark

## Requirements

* Python >= 3.8
* PyTorch >= 1.12.0

## Support Features

### Pipelines

* Single MARL training
* Perturbation-based attacks
* Adversarial traitors

### Algorithms

* MAPPO
* MADDPG
* Iterative Gradient Sign Perturbations

### Environments

* SMAC
* SMACv2
* Multi-Agent MuJoCo
* PettingZoo MPE
* Google Research Football
* Gym
* Toy Example
* Custom environments

## Usage Examples

### Single Algorithm Training

```bash
python -u train.py --env <env_name> --algo <algo_name> --exp_name <exp_name> --run single
```

### Perturbation-based Attack

```bash
python -u train.py --env <env_name> --algo <perturbation_name> --exp_name <exp_name> --run perturbation --victim <victim_algo_name> --victim.model_dir <dir/to/your/model>
```

### Adversarial Traitors

```bash
python -u train.py --env <env_name> --algo <traitor_algo_name> --exp_name <exp_name> --run traitor --victim <victim_algo_name> --victim.model_dir <dir/to/your/model>
```

### Load Victim Config fron JSON

```bash
# It will load environment and victim configurations from JSON, together with the victim's checkpoints in "models/" directory
python -u train.py --algo <adv_algo_name> --exp_name <exp_name> --run [traitor|perturbation] --load_victim <dir/to/victim/results>
```

### Argument Decided Parameters

```bash
# Here is an example. "A.B" -> A means namespace (algo, env, victim), and B means the parameter name.
python -u train.py --algo.lr 5e-4 --victim.share_param False --env.map_name 2s3z
```