import os
from .nni.rlplotter.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tblog_to_csv(log_path, ytag="env/eval_return_mean", exp_name="mappo_iclr", env_name="halfcheetah_6x1"):
    event_data = EventAccumulator(log_path)
    event_data.Reload()
    print(event_data.scalars)
    reward = event_data.scalars.Items(ytag)
    logger = Logger(log_dir=env_name, exp_name=exp_name)
    for elm in reward:
        logger.update(score=[elm.value], total_steps=elm.step)


tblog_to_csv("/root/nni-experiments/qpwvbx27/trials/AO3Er/tensorboard/logs", exp_name="exp_name", env_name="env_name")