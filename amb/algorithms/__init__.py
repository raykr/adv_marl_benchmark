from amb.algorithms.maddpg import MADDPG
from amb.algorithms.mappo import MAPPO

ALGO_REGISTRY = {
    "maddpg": MADDPG,
    "mappo": MAPPO
}