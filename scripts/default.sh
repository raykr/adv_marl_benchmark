
# - [x] MAPPO, SMAC, 3m
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name default
# - [x] MAPPO, SMAC, 3s_vs_5z
# python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name default

# - [ ] MADDPG, SMAC, 3m， 超过500K步后训崩
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --algo.num_env_steps 5000000 --exp_name default
# - [ ] MADDPG, SMAC, 3s_vs_5z
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --algo.num_env_steps 10000000 --exp_name default

# - [x] QMIX, SMAX, 3m
# python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name default
# - [ ] QMIX, SMAC, 3s_vs_5z
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo qmix --run single --algo.num_env_steps 10000000 --exp_name default


# - [ ] MAPPO, MPE, simple_spread_v3
# python -u ../single_train.py --env pettingzoo_mpe --env.scenario simple_spread_v3 --algo mappo --run single --algo.num_env_steps 10000000 --exp_name default
# - [ ] MAPPO, MPE, simple_speaker_listener_v4
# python -u ../single_train.py --env pettingzoo_mpe --env.scenario simple_speaker_listener_v4 --algo mappo --run single --algo.num_env_steps 10000000 --exp_name default
# - [ ] MADDPG, MPE, simple_spread_v3
# python -u ../single_train.py --env pettingzoo_mpe --env.scenario simple_spread_v3 --algo maddpg --run single --algo.num_env_steps 10000000 --exp_name default
# - [ ] MADDPG, MPE, simple_speaker_listener_v4
# python -u ../single_train.py --env pettingzoo_mpe --env.scenario simple_speaker_listener_v4 --algo maddpg --run single --algo.num_env_steps 10000000 --exp_name default

# - [ ] MAPPO, MAMuJoCo, HalfCheetah-6x1
# python -u ../single_train.py --env mamujoco --env.scenario HalfCheetah --env.agent_conf "6x1" --algo mappo --run single --algo.num_env_steps 10000000 --exp_name default
# - [ ] MAPPO, MAMuJoCo, Hopper-3x1
# python -u ../single_train.py --env mamujoco --env.scenario Hopper --env.agent_conf "3x1" --algo mappo --run single --algo.num_env_steps 10000000 --exp_name default
# - [ ] MADDPG, MAMuJoCo, HalfCheetah-6x1
# python -u ../single_train.py --env mamujoco --env.scenario HalfCheetah --env.agent_conf "6x1" --algo maddpg --run single --algo.num_env_steps 10000000 --exp_name default
# - [ ] MADDPG, MAMuJoCo, Hopper-3x1
# python -u ../single_train.py --env mamujoco --env.scenario Hopper --env.agent_conf "3x1" --algo maddpg --run single --algo.num_env_steps 10000000 --exp_name default