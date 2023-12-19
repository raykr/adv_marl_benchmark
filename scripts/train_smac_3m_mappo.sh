python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name default > logs/smac/3m/mappo/default/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name entropy_coef_0.0001  --algo.entropy_coef 0.0001 > logs/smac/3m/mappo/entropy_coef_0.0001/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name entropy_coef_0.001  --algo.entropy_coef 0.001 > logs/smac/3m/mappo/entropy_coef_0.001/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name entropy_coef_0.1  --algo.entropy_coef 0.1 > logs/smac/3m/mappo/entropy_coef_0.1/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name entropy_coef_0.5  --algo.entropy_coef 0.5 > logs/smac/3m/mappo/entropy_coef_0.5/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name entropy_coef_1.0  --algo.entropy_coef 1.0 > logs/smac/3m/mappo/entropy_coef_1.0/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name gamma_0.95  --algo.gamma 0.95 > logs/smac/3m/mappo/gamma_0.95/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name gamma_1  --algo.gamma 1 > logs/smac/3m/mappo/gamma_1/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name hidden_sizes_64_64  --algo.hidden_sizes "[64, 64]" > logs/smac/3m/mappo/hidden_sizes_64_64/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name hidden_sizes_512_512  --algo.hidden_sizes "[512, 512]" > logs/smac/3m/mappo/hidden_sizes_512_512/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name activation_func_leaky_relu  --algo.activation_func leaky_relu > logs/smac/3m/mappo/activation_func_leaky_relu/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name activation_func_selu  --algo.activation_func selu > logs/smac/3m/mappo/activation_func_selu/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name activation_func_sigmoid  --algo.activation_func sigmoid > logs/smac/3m/mappo/activation_func_sigmoid/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name activation_func_tanh  --algo.activation_func tanh > logs/smac/3m/mappo/activation_func_tanh/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name initialization_method_xavier_uniform_  --algo.initialization_method xavier_uniform_ > logs/smac/3m/mappo/initialization_method_xavier_uniform_/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name use_recurrent_policy_True  --algo.use_recurrent_policy True > logs/smac/3m/mappo/use_recurrent_policy_True/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name use_feature_normalization_False  --algo.use_feature_normalization False > logs/smac/3m/mappo/use_feature_normalization_False/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name lr_5e-05  --algo.lr 5e-05 > logs/smac/3m/mappo/lr_5e-05/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name lr_0.005  --algo.lr 0.005 > logs/smac/3m/mappo/lr_0.005/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name critic_lr_5e-05  --algo.critic_lr 5e-05 > logs/smac/3m/mappo/critic_lr_5e-05/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name critic_lr_0.005  --algo.critic_lr 0.005 > logs/smac/3m/mappo/critic_lr_0.005/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name use_gae_False  --algo.use_gae False > logs/smac/3m/mappo/use_gae_False/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name use_popart_False  --algo.use_popart False > logs/smac/3m/mappo/use_popart_False/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/mappo.json --exp_name share_param_False  --algo.share_param False > logs/smac/3m/mappo/share_param_False/train.log 2>&1
