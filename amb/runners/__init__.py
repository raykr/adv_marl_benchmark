def get_runner(run, algo):
    if run == "single":
        if algo == "mappo":
            from amb.runners.single.on_policy_runner import OnPolicyRunner
            return OnPolicyRunner
        elif algo == "maddpg" or algo == "qmix" or algo == "vdn" or algo == "iql" or algo == "qtran" or algo == "coma":
            from amb.runners.single.off_policy_runner import OffPolicyRunner
            return OffPolicyRunner
    if run == "perturbation":
        from amb.runners.perturbation.base_runner import BaseRunner
        return BaseRunner
    if run == "traitor":
        if algo == "mappo":
            from amb.runners.traitor.on_policy_runner import OnPolicyRunner
            return OnPolicyRunner
        elif algo == "maddpg":
            from amb.runners.traitor.off_policy_runner import OffPolicyRunner
            return OffPolicyRunner
    if run == "dual":
        if algo == "mappo":
            from amb.runners.dual.on_policy_runner import OnPolicyRunner
            return OnPolicyRunner
        elif algo == "maddpg":
            from amb.runners.dual.off_policy_runner import OffPolicyRunner
            return OffPolicyRunner