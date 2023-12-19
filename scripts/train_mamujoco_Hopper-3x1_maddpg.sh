python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name default > logs/mamujoco/Hopper-3x1/maddpg/default/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name expl_noise_0.001  --algo.expl_noise 0.001 > logs/mamujoco/Hopper-3x1/maddpg/expl_noise_0.001/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name expl_noise_0.01  --algo.expl_noise 0.01 > logs/mamujoco/Hopper-3x1/maddpg/expl_noise_0.01/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name expl_noise_0.5  --algo.expl_noise 0.5 > logs/mamujoco/Hopper-3x1/maddpg/expl_noise_0.5/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name expl_noise_1.0  --algo.expl_noise 1.0 > logs/mamujoco/Hopper-3x1/maddpg/expl_noise_1.0/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name gamma_0.95  --algo.gamma 0.95 > logs/mamujoco/Hopper-3x1/maddpg/gamma_0.95/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name gamma_1  --algo.gamma 1 > logs/mamujoco/Hopper-3x1/maddpg/gamma_1/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name hidden_sizes_64_64  --algo.hidden_sizes "[64, 64]" > logs/mamujoco/Hopper-3x1/maddpg/hidden_sizes_64_64/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name hidden_sizes_512_512  --algo.hidden_sizes "[512, 512]" > logs/mamujoco/Hopper-3x1/maddpg/hidden_sizes_512_512/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name activation_func_leaky_relu  --algo.activation_func leaky_relu > logs/mamujoco/Hopper-3x1/maddpg/activation_func_leaky_relu/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name activation_func_selu  --algo.activation_func selu > logs/mamujoco/Hopper-3x1/maddpg/activation_func_selu/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name activation_func_sigmoid  --algo.activation_func sigmoid > logs/mamujoco/Hopper-3x1/maddpg/activation_func_sigmoid/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name activation_func_tanh  --algo.activation_func tanh > logs/mamujoco/Hopper-3x1/maddpg/activation_func_tanh/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name initialization_method_xavier_uniform_  --algo.initialization_method xavier_uniform_ > logs/mamujoco/Hopper-3x1/maddpg/initialization_method_xavier_uniform_/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name use_recurrent_policy_True  --algo.use_recurrent_policy True > logs/mamujoco/Hopper-3x1/maddpg/use_recurrent_policy_True/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name use_feature_normalization_False  --algo.use_feature_normalization False > logs/mamujoco/Hopper-3x1/maddpg/use_feature_normalization_False/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name lr_5e-05  --algo.lr 5e-05 > logs/mamujoco/Hopper-3x1/maddpg/lr_5e-05/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name lr_0.005  --algo.lr 0.005 > logs/mamujoco/Hopper-3x1/maddpg/lr_0.005/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name critic_lr_5e-05  --algo.critic_lr 5e-05 > logs/mamujoco/Hopper-3x1/maddpg/critic_lr_5e-05/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name critic_lr_0.005  --algo.critic_lr 0.005 > logs/mamujoco/Hopper-3x1/maddpg/critic_lr_0.005/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name n_step_10  --algo.n_step 10 > logs/mamujoco/Hopper-3x1/maddpg/n_step_10/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name n_step_50  --algo.n_step 50 > logs/mamujoco/Hopper-3x1/maddpg/n_step_50/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name share_param_False  --algo.share_param False > logs/mamujoco/Hopper-3x1/maddpg/share_param_False/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name batch_size_500  --algo.batch_size 500 > logs/mamujoco/Hopper-3x1/maddpg/batch_size_500/train.log 2>&1
python -u ../single_train.py --load_config settings/mamujoco/Hopper-3x1/maddpg.json --exp_name batch_size_5000  --algo.batch_size 5000 > logs/mamujoco/Hopper-3x1/maddpg/batch_size_5000/train.log 2>&1
