import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import math
#import tensorflow as tf
#from gym.envs.mujoco import mujoco_py.MjSim

class UruhlEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'uruhl.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        # Carry out one step
        # Don't forget to do self.do_simulation(a, self.frame_skip)
        self.do_simulation(a, self.frame_skip)
        #state = self.state_vector()
        #notdone = np.isfinite(state).all() \
        #    and state[2] >= 0.2 and state[2] <= 1.0
        #done = not notdone
        done = False
        ob = self._get_obs()
        angle = self._get_angle()
        #if (angle>70 or angle<-70):             #Radian or Degree???
    #       done = True
        reward = self._get_reward(done)
        return ob, reward, done,1

    def _get_angle(self):
        qu = self.data.xquat[1]
        #qu = self.data.xmat[1]
        X,Y,Z = self.quaternion_to_euler_angle(qu[1],qu[2],qu[3],qu[0])
        print [X,Y,Z]
        return [X,Y,Z]
        #print qu
        #qu = qu.reshape(3,3)
        #x = self.rotationMatrixToEulerAngles(qu)*(180/math.pi)
        #print x
        #return x

    def _get_reward(self,done):
        if done == True:
            return -5
        else:
            return 5

    def _get_obs(self):
        # Observation of environment feed to agent. This should never be called
        # directly but should be returned through reset_model and step
        obse = self.get_sensor_sensordata()
        #print obse
        #print self.self_dat()
        #print self.model.__dict__
        #print self.get_body_com("mainframe")[0]
        #print self.data.xquat[1]
        #quaternion = self.data.xquat[1]
        #euler = tf.transformations.euler_from_quaternion(quaternion)
        #print euler
        return obse

    def reset_model(self):
        # Reset model to original state.
        # This is called in the overall env.reset method
        # do not call this method directly.
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
        #qvel = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8

    def quaternion_to_euler_angle(self,w, y,x,z):
        ysqr = y * y

    	t0 = +2.0 * (w * x + y * z)
    	t1 = +1.0 - 2.0 * (x * x + ysqr)
    	X = math.degrees(math.atan2(t0, t1))

    	t2 = +2.0 * (w * y - z * x)
    	t2 = +1.0 if t2 > +1.0 else t2
    	t2 = -1.0 if t2 < -1.0 else t2
    	Y = math.degrees(math.asin(t2))

    	t3 = +2.0 * (w * z + x * y)
    	t4 = +1.0 - 2.0 * (ysqr + z * z)
    	Z = math.degrees(math.atan2(t3, t4))

    	return X, Y, Z

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self,R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self,R) :
        assert(self.isRotationMatrix(R))

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])










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
"""
///////////////////////////////
 Quaternion to Euler
///////////////////////////////
enum RotSeq{zyx, zyz, zxy, zxz, yxz, yxy, yzx, yzy, xyz, xyx, xzy,xzx};

def twoaxisrot(double r11, double r12, double r21, double r31, double r32, double res[]):
    res[0] = atan2( r11, r12 )
    res[1] = acos ( r21 )
    res[2] = atan2( r31, r32 )

def threeaxisrot(double r11, double r12, double r21, double r31, double r32, double res[]):
    res[0] = atan2( r31, r32 )
    res[1] = asin ( r21 )
    res[2] = atan2( r11, r12 )

def quaternion2Euler(const Quaternion& q, double res[], RotSeq rotSeq):
    def switch(rotSeq):
        switcher={
        zyx: threeaxisrot( 2*(q.x*q.y + q.w*q.z),q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z,-2*(q.x*q.z - q.w*q.y),2*(q.y*q.z + q.w*q.x),q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z,res),
        zyz: twoaxisrot( 2*(q.y*q.z - q.w*q.x),2*(q.x*q.z + q.w*q.y),q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z,2*(q.y*q.z + q.w*q.x),-2*(q.x*q.z - q.w*q.y),res),
        zxy: threeaxisrot( -2*(q.x*q.y - q.w*q.z),q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z,2*(q.y*q.z + q.w*q.x),-2*(q.x*q.z - q.w*q.y),q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z,res),
        zxz: twoaxisrot( 2*(q.x*q.z + q.w*q.y),-2*(q.y*q.z - q.w*q.x),q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z,2*(q.x*q.z - q.w*q.y),2*(q.y*q.z + q.w*q.x),res),
        yxz: threeaxisrot( 2*(q.x*q.z + q.w*q.y),q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z,-2*(q.y*q.z - q.w*q.x),2*(q.x*q.y + q.w*q.z),q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z,res),
        yzx: threeaxisrot( -2*(q.x*q.z - q.w*q.y),q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z,2*(q.x*q.y + q.w*q.z),-2*(q.y*q.z - q.w*q.x),q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z,res),
        xyz: threeaxisrot( -2*(q.y*q.z - q.w*q.x),q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z,2*(q.x*q.z + q.w*q.y),-2*(q.x*q.y - q.w*q.z),q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z,res),
        xzy: threeaxisrot( 2*(q.y*q.z + q.w*q.x),q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z,-2*(q.x*q.y - q.w*q.z),2*(q.x*q.z + q.w*q.y),q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z,res),
            }
    func = switcher.get(argument, lambda: "Invalid month") """
