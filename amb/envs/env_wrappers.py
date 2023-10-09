"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
import torch
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
import copy


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """

    closed = False
    viewer = None

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self, num_envs, observation_space, share_observation_space, action_space
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions, filled=None):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self, filled=None):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions, filled=None):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions, filled)
        return self.step_wait(filled)

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError


def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if "bool" in done.__class__.__name__:  # done is a bool
                if done:  # if done, save the original obs, state, and available actions in info, and then reset
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):  # if done, save the original obs, state, and available actions in info, and then reset
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "step_dual":
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if "bool" in done[0].__class__.__name__:  # done is a bool
                if done[0]:  # if done, save the original obs, state, and available actions in info, and then reset
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done[0]):  # if done, save the original obs, state, and available actions in info, and then reset
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == "reset":
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        # elif cmd == "reset_task":
        #     ob = env.reset_task()
        #     remote.send(ob)
        elif cmd == "render":
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space)
            )
        # elif cmd == "render_vulnerability":
        #     fr = env.render_vulnerability(data)
        #     remote.send((fr))
        elif cmd == "get_num_agents":
            remote.send((env.n_agents))
        elif cmd == "get_num_agents_dual":
            remote.send((env.n_angels, env.n_demons))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=shareworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_num_agents", None))
        self.n_agents = self.remotes[0].recv()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space
        )

    def step_async(self, actions, filled=None):
        if filled is None:
            for remote, action in zip(self.remotes, actions):
                remote.send(("step", action))
        else:
            for remote, action, fill in zip(self.remotes, actions, filled):
                if fill:
                    remote.send(("step", action))
        self.waiting = True

    def step_wait(self, filled=None):
        if filled is None:
            results = [remote.recv() for remote in self.remotes]
        else:
            results = []
            for i in range(len(filled)):
                if filled[i]:
                    results.append(self.remotes[i].recv())
                else:
                    results.append(None)
            for elm in results:
                if elm is not None:
                    temp = elm
                    break
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = copy.deepcopy(elm)
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return (
            np.stack(obs),
            np.stack(share_obs),
            np.stack(rews),
            np.stack(dones),
            infos,
            np.stack(available_actions),
        )

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    # def reset_task(self):
    #     for remote in self.remotes:
    #         remote.send(("reset_task", None))
    #     return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


class ShareSubprocVecDualEnv(ShareSubprocVecEnv):
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=shareworker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            # p.daemon = (
            #     True  # if the main process crashes, we should not cause things to hang
            # )
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("get_num_agents", None))
        self.n_agents = self.remotes[0].recv()
        self.remotes[0].send(("get_spaces", None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(("get_num_agents_dual", None))
        self.n_angels, self.n_demons = self.remotes[0].recv()
        ShareVecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space
        )
        

    def step_async(self, actions, filled=None):
        angel_actions, demon_actions = actions
        if filled is None:
            for remote, a_action, d_action in zip(self.remotes, angel_actions, demon_actions):
                remote.send(("step_dual", (a_action, d_action)))
        else:
            for remote, a_action, d_action, fill in zip(self.remotes, angel_actions, demon_actions, filled):
                if fill:
                    remote.send(("step_dual", (a_action, d_action)))
        self.waiting = True

    def step_wait(self, filled=None):
        if filled is None:
            results = [remote.recv() for remote in self.remotes]
        else:
            results = []
            for i in range(len(filled)):
                if filled[i]:
                    results.append(self.remotes[i].recv())
                else:
                    results.append(None)
            for elm in results:
                if elm is not None:
                    temp = elm
                    break
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = copy.deepcopy(elm)
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        
        return (
            [np.stack(o) for o in zip(*obs)],
            [np.stack(o) for o in zip(*share_obs)],
            [np.stack(o) for o in zip(*rews)],
            [np.stack(o) for o in zip(*dones)],
            [o for o in zip(*infos)],
            [np.stack(o) for o in zip(*available_actions)],
        )
    
    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return (
            [np.stack(o) for o in zip(*obs)],
            [np.stack(o) for o in zip(*share_obs)],
            [np.stack(o) for o in zip(*available_actions)],
        )

# single env
class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self,
            len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space,
        )
        self.actions = None
        try:
            self.n_agents = env.n_agents
        except:
            pass

        try:
            self.n_angels = env.n_angels
            self.n_demons = env.n_demons
        except:
            pass

    def step_async(self, actions, filled=None):
        self.actions = actions

    def step_wait(self, filled=None):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results)
        )

        for i, done in enumerate(dones):
            if "bool" in done.__class__.__name__:  # done is a bool
                if done:  # if done, save the original obs, state, and available actions in info, and then reset
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(done):  # if done, save the original obs, state, and available actions in info, and then reset
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


class ShareDummyVecDualEnv(ShareDummyVecEnv):
    def step_wait(self, filled=None):
        angel_actions, demon_actions = self.actions
        results = [env.step((a, d)) for (a, d, env) in zip(angel_actions, demon_actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)

        obs, share_obs, available_actions = list(obs), list(share_obs), list(available_actions)

        for i, done in enumerate(dones):
            if "bool" in done[0].__class__.__name__:  # done is a bool
                if done[0]:  # if done, save the original obs, state, and available actions in info, and then reset
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(done[0]):  # if done, save the original obs, state, and available actions in info, and then reset
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()

        self.actions = None

        return (
            [np.stack(o) for o in zip(*obs)],
            [np.stack(o) for o in zip(*share_obs)],
            [np.stack(o) for o in zip(*rews)],
            [np.stack(o) for o in zip(*dones)],
            [o for o in zip(*infos)],
            [np.stack(o) for o in zip(*available_actions)],
        )
    
    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = zip(*results)

        return (
            [np.stack(o) for o in zip(*obs)],
            [np.stack(o) for o in zip(*share_obs)],
            [np.stack(o) for o in zip(*available_actions)],
        )