from amb.runners.single.off_policy_runner import OffPolicyRunner
from amb.runners.single.on_policy_runner import OnPolicyRunner

RUNNER_REGISTRY = {
    "mappo": OnPolicyRunner,
    "maddpg": OffPolicyRunner,
}