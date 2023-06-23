from amb.runners.single.off_policy_runner import OffPolicyRunner
from amb.runners.single.on_policy_runner import OnPolicyRunner
from amb.runners.perturbation.base_runner import BaseRunner as PerturbationRunner

RUNNER_REGISTRY = {
    "mappo": OnPolicyRunner,
    "maddpg": OffPolicyRunner,
    "perturbation": PerturbationRunner,
}