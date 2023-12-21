python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name default > logs/smac/3m/qmix/default/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name epsilon_anneal_time_50000  --algo.epsilon_anneal_time 50000 > logs/smac/3m/qmix/epsilon_anneal_time_50000/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name epsilon_anneal_time_200000  --algo.epsilon_anneal_time 200000 > logs/smac/3m/qmix/epsilon_anneal_time_200000/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name epsilon_finish_0.01  --algo.epsilon_finish 0.01 > logs/smac/3m/qmix/epsilon_finish_0.01/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name epsilon_finish_0.1  --algo.epsilon_finish 0.1 > logs/smac/3m/qmix/epsilon_finish_0.1/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name eps_delta_l  --algo.epsilon_anneal_time 80000 --algo.epsilon_finish 0.24 > logs/smac/3m/qmix/eps_delta_l/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name eps_delta_r  --algo.epsilon_anneal_time 104211 --algo.epsilon_finish 0.01 > logs/smac/3m/qmix/eps_delta_r/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name gamma_0.95  --algo.gamma 0.95 > logs/smac/3m/qmix/gamma_0.95/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name gamma_1.0  --algo.gamma 1.0 > logs/smac/3m/qmix/gamma_1.0/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name hidden_sizes_64_64  --algo.hidden_sizes "[64, 64]" > logs/smac/3m/qmix/hidden_sizes_64_64/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name hidden_sizes_256_256  --algo.hidden_sizes "[256, 256]" > logs/smac/3m/qmix/hidden_sizes_256_256/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name activation_func_leaky_relu  --algo.activation_func leaky_relu > logs/smac/3m/qmix/activation_func_leaky_relu/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name activation_func_selu  --algo.activation_func selu > logs/smac/3m/qmix/activation_func_selu/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name activation_func_sigmoid  --algo.activation_func sigmoid > logs/smac/3m/qmix/activation_func_sigmoid/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name activation_func_tanh  --algo.activation_func tanh > logs/smac/3m/qmix/activation_func_tanh/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name initialization_method_xavier_uniform_  --algo.initialization_method xavier_uniform_ > logs/smac/3m/qmix/initialization_method_xavier_uniform_/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name use_recurrent_policy_True  --algo.use_recurrent_policy True > logs/smac/3m/qmix/use_recurrent_policy_True/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name use_feature_normalization_False  --algo.use_feature_normalization False > logs/smac/3m/qmix/use_feature_normalization_False/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name lr_5e-05  --algo.lr 5e-05 > logs/smac/3m/qmix/lr_5e-05/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name lr_0.005  --algo.lr 0.005 > logs/smac/3m/qmix/lr_0.005/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name n_step_5  --algo.n_step 5 > logs/smac/3m/qmix/n_step_5/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name n_step_50  --algo.n_step 50 > logs/smac/3m/qmix/n_step_50/train.log 2>&1
python -u ../single_train.py --load_config settings/smac/3m/qmix.json --exp_name share_param_False  --algo.share_param False > logs/smac/3m/qmix/share_param_False/train.log 2>&1
