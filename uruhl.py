import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
#from gym.envs.mujoco import mujoco_py.MjSim

class UruhlEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'uruhl.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        # Carry out one step
        # Don't forget to do self.do_simulation(a, self.frame_skip)
        xposbefore = self.get_body_com("mainframe")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("mainframe")[0]
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1)))


        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0

        done = not notdone
        ob = self._get_obs()
    #   angle = _get_angle(ob)
    #   if (angle>70 or angle<-70):             #Radian or DEgree???
    #       done = True
        reward = self._get_reward(done)
        return ob, reward, done,1
#    def _get_angle(self,ob):
#        ax,ay,az,gx,gy,gx,_,_ = ob #I donno whether they are in the correct cordinate system
    def _get_reward(self,done):
        if done == True:
            return -5
        else:
            return 5

    def _get_obs(self):
        # Observation of environment feed to agent. This should never be called
        # directly but should be returned through reset_model and step
        acc = self.get_sensor_sensordata("Acc")
        print acc
        return acc
        # print acc
        # return np.concatenate([
        #     self.model.data.qpos.flat[2:],
        #     self.model.data.qvel.flat,
        #     np.clip(self.model.data.cfrc_ext, -1, 1).flat,
        # ])

    def reset_model(self):
        # Reset model to original state.
        # This is called in the overall env.reset method
        # do not call this method directly.
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        #qvel = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8

"""
    def __init__(self):
        pass

    def _step(self, action):
        //

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.

        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {}

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        pass

    def _get_reward(self):
         Reward is given for XY.
        if self.status == FOOBAR:
            return 1
        elif self.status == ABC:
            return self.somestate ** 2
        else:
            return 0"""
