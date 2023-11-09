# trick 只跑正常训练的模型

# samc 3m mappo 
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_default
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_entropy_coef_0.0001 --algo.entropy_coef 0.0001
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_entropy_coef_0.001 --algo.entropy_coef 0.001
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_entropy_coef_0.1 --algo.entropy_coef 0.1
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_entropy_coef_0.5 --algo.entropy_coef 0.5
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_entropy_coef_1 --algo.entropy_coef 1
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_hidden_size_64_64 --algo.hidden_sizes "[64, 64]"
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_hidden_size_256_256 --algo.hidden_sizes "[256, 256]"
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_activate_sigmoid --algo.activation_func sigmoid
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_activate_tanh --algo.activation_func tanh
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_activate_leaky_relu --algo.activation_func leaky_relu
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_activate_selu --algo.activation_func selu
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_init_uniform --algo.initialization_method xavier_uniform_
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_rnn_true --algo.use_recurrent_policy True
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_norm_false --algo.use_feature_normalization False
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_lr_0.00005 --algo.lr 0.00005
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_lr_0.005 --algo.lr 0.005
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_lr_0.05 --algo.lr 0.05
python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_critic_lr_0.00005 --algo.critic_lr 0.00005
python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_critic_lr_0.005 --algo.critic_lr 0.005
python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_critic_lr_0.05 --algo.critic_lr 0.05
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_share_false --algo.share_param False
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_gamma_0.9 --algo.gamma 0.9
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_gamma_0.95 --algo.gamma 0.95
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_gae_false --algo.use_gae False
# python -u ../single_train.py --env smac --env.map_name 3m --algo mappo --run single --algo.num_env_steps 5000000 --exp_name mappo_smac_3m_popart_false --algo.use_popart False

# samc 3s_vs_5z mappo 
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_default
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_entropy_coef_0.0001 --algo.entropy_coef 0.0001
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_entropy_coef_0.001 --algo.entropy_coef 0.001
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_entropy_coef_0.1 --algo.entropy_coef 0.1
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_entropy_coef_0.5 --algo.entropy_coef 0.5
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_entropy_coef_1 --algo.entropy_coef 1
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_hidden_size_64_64 --algo.hidden_sizes "[64, 64]"
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_hidden_size_256_256 --algo.hidden_sizes "[256, 256]"
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_activate_sigmoid --algo.activation_func sigmoid
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_activate_tanh --algo.activation_func tanh
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_activate_leaky_relu --algo.activation_func leaky_relu
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_activate_selu --algo.activation_func selu
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_init_uniform --algo.initialization_method xavier_uniform_
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_rnn_true --algo.use_recurrent_policy True
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_norm_false --algo.use_feature_normalization False
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_lr_0.00005 --algo.lr 0.00005
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_lr_0.005 --algo.lr 0.005
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_lr_0.05 --algo.lr 0.05
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_critic_lr_0.00005 --algo.critic_lr 0.00005
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_critic_lr_0.005 --algo.critic_lr 0.005
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_critic_lr_0.05 --algo.critic_lr 0.05
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_share_false --algo.share_param False
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_gamma_0.9 --algo.gamma 0.9
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_gamma_0.95 --algo.gamma 0.95
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_gae_false --algo.use_gae False
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo mappo --run single --algo.num_env_steps 10000000 --exp_name mappo_smac_3s_vs_5z_popart_false --algo.use_popart False



