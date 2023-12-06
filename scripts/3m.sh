# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name default
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name entropycoef_0.0001 --algo.entropy_coef 0.0001
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name entropycoef_0.001 --algo.entropy_coef 0.001
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name entropycoef_0.1 --algo.entropy_coef 0.1
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name entropycoef_0.5 --algo.entropy_coef 0.5
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name entropycoef_1 --algo.entropy_coef 1
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name hiddensize_64_64 --algo.hidden_sizes "[64, 64]"
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name hiddensize_256_256 --algo.hidden_sizes "[256, 256]"
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name activate_sigmoid --algo.activation_func sigmoid
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name activate_tanh --algo.activation_func tanh
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name activate_leaky_relu --algo.activation_func leaky_relu
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name activate_selu --algo.activation_func selu
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name init_uniform --algo.initialization_method xavier_uniform_
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name rnn_true --algo.use_recurrent_policy True
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name norm_false --algo.use_feature_normalization False
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name lr_0.00005 --algo.lr 0.00005
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name lr_0.005 --algo.lr 0.005
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name lr_0.05 --algo.lr 0.05
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name criticlr_0.00005 --algo.critic_lr 0.00005
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name criticlr_0.005 --algo.critic_lr 0.005
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name criticlr_0.05 --algo.critic_lr 0.05
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name share_false --algo.share_param False
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name gamma_0.9 --algo.gamma 0.9
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name gamma_0.95 --algo.gamma 0.95
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name gae_false --algo.use_gae False
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name popart_false --algo.use_popart False

# python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name default
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name eps_anneal_time_50000 --algo.eps_anneal_time 50000
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name eps_anneal_time_200000 --algo.eps_anneal_time 200000
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name eps_finish_0.01 --algo.eps_finish 0.01
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name eps_finish_0.1 --algo.eps_finish 0.1
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name eps_delta_l --algo.eps_anneal_time 80000 --algo.eps_finish 0.24
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name eps_delta_r --algo.eps_anneal_time 104211 --algo.eps_finish 0.01
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name hiddensize_64_64 --algo.hidden_sizes "[64, 64]"
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name hiddensize_256_256 --algo.hidden_sizes "[256, 256]"
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name activate_sigmoid --algo.activation_func sigmoid
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name activate_tanh --algo.activation_func tanh
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name activate_leaky_relu --algo.activation_func leaky_relu
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name activate_selu --algo.activation_func selu
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name init_uniform --algo.initialization_method xavier_uniform_
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name rnn_true --algo.use_recurrent_policy True
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name norm_false --algo.use_feature_normalization False
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name lr_0.00005 --algo.lr 0.00005
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name lr_0.005 --algo.lr 0.005
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name lr_0.05 --algo.lr 0.05
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name criticlr_0.00005 --algo.critic_lr 0.00005
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name criticlr_0.005 --algo.critic_lr 0.005
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name criticlr_0.05 --algo.critic_lr 0.05
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name share_false --algo.share_param False
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name share_true --algo.share_param True
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name gamma_0.9 --algo.gamma 0.9
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name gamma_0.95 --algo.gamma 0.95
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name gae_true --algo.use_gae True
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name nstep_5 --algo.n_step 5
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name nstep_10 --algo.n_step 10
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name nstep_20 --algo.n_step 20
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name batch_100 --algo.batch_size 100
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name batch_500 --algo.batch_size 500
python -u ../single_train.py --env smac --env.map_name 3m --algo qmix --run single --algo.num_env_steps 5000000 --exp_name batch_5000 --algo.batch_size 5000