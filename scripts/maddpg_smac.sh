# trick 只跑正常训练的模型

# samc 3m maddpg 
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_default
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_expl_noise_0.0001 --algo.expl_noise 0.001
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_expl_noise_0.001 --algo.expl_noise 0.01
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_expl_noise_0.5 --algo.expl_noise 0.5
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_expl_noise_1 --algo.expl_noise 1
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_hidden_size_64_64 --algo.hidden_sizes "[64, 64]"
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_hidden_size_256_256 --algo.hidden_sizes "[256, 256]"
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_activate_sigmoid --algo.activation_func sigmoid
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_activate_tanh --algo.activation_func tanh
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_activate_leaky_relu --algo.activation_func leaky_relu
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_activate_selu --algo.activation_func selu
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_init_uniform --algo.initialization_method xavier_uniform_
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_rnn_true --algo.use_recurrent_policy True
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_norm_false --algo.use_feature_normalization False
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_lr_0.00005 --algo.lr 0.00005
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_lr_0.005 --algo.lr 0.005
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_lr_0.05 --algo.lr 0.05
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_critic_lr_0.00005 --algo.critic_lr 0.00005
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_critic_lr_0.005 --algo.critic_lr 0.005
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_critic_lr_0.05 --algo.critic_lr 0.05
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_share_false --algo.share_param False
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_gamma_0.9 --algo.gamma 0.9
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_gamma_0.95 --algo.gamma 0.95
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_gae_false --algo.use_gae False
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_nstep_2 --algo.n_step 2
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_nstep_5 --algo.n_step 5
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_nstep_10 --algo.n_step 10
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_batch_100 --algo.batch_size 100
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_batch_500 --algo.batch_size 500
python -u ../single_train.py --env smac --env.map_name 3m --algo maddpg --run single --exp_name maddpg_smac_3m_batch_5000 --algo.batch_size 5000

# samc 3s_vs_5z maddpg 
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_default
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_expl_noise_0.0001 --algo.expl_noise 0.0001
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_expl_noise_0.001 --algo.expl_noise 0.001
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_expl_noise_0.1 --algo.expl_noise 0.1
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_expl_noise_0.5 --algo.expl_noise 0.5
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_expl_noise_1 --algo.expl_noise 1
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_hidden_size_64_64 --algo.hidden_sizes "[64, 64]"
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_hidden_size_256_256 --algo.hidden_sizes "[256, 256]"
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_activate_sigmoid --algo.activation_func sigmoid
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_activate_tanh --algo.activation_func tanh
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_activate_leaky_relu --algo.activation_func leaky_relu
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_activate_selu --algo.activation_func selu
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_init_uniform --algo.initialization_method xavier_uniform_
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_rnn_true --algo.use_recurrent_policy True
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_norm_false --algo.use_feature_normalization False
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_lr_0.00005 --algo.lr 0.00005
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_lr_0.005 --algo.lr 0.005
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_lr_0.05 --algo.lr 0.05
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_critic_lr_0.00005 --algo.critic_lr 0.00005
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_critic_lr_0.005 --algo.critic_lr 0.005
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_critic_lr_0.05 --algo.critic_lr 0.05
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_share_false --algo.share_param False
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_gamma_0.9 --algo.gamma 0.9
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_gamma_0.95 --algo.gamma 0.95
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_gae_false --algo.use_gae False
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_nstep_2 --algo.n_step 2
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_nstep_5 --algo.n_step 5
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_nstep_10 --algo.n_step 10
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_batch_100 --algo.batch_size 100
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_batch_500 --algo.batch_size 500
python -u ../single_train.py --env smac --env.map_name 3s_vs_5z --algo maddpg --run single --exp_name maddpg_smac_3s_vs_5z_batch_5000 --algo.batch_size 5000



