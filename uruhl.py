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
        self.do_simulation(a, self.frame_skip)
        done = False
        ob = self._get_obs()
        angle = self._get_angle()
        if (abs(angle)>60):
            done = True
        reward = self._get_reward(angle,done)
        return ob, reward, done,self.get_sensor_sensordata()[4]

    def _get_angle(self):
        qu = self.data.xquat[1]
        #qu = self.data.xmat[1]
        X,Y,Z = self.quaternion_to_euler_angle(qu[1],qu[2],qu[3],qu[0])
        #print X
        return X
        #print qu
        #qu = qu.reshape(3,3)
        #x = self.rotationMatrixToEulerAngles(qu)*(180/math.pi)
        #print x
        #return x

    def _get_reward(self,angle,done):
        if done == True:
            return -1
        elif abs(angle)<20:
            return 4
        else:
            return 1



    def _get_obs(self):
        # Observation of environment feed to agent. This should never be called
        # directly but should be returned through reset_model and step
        #obse = self.get_sensor_sensordata()
        #print obse
        #print self.self_dat()
        #print self.model.__dict__
        #print self.get_body_com("mainframe")[0]
        #print self.data.xquat[1]
        #quaternion = self.data.xquat[1]
        #euler = tf.transformations.euler_from_quaternion(quaternion)
        #print euler
        #ob[6] = (ob[6]/0.7)%180
        #return obse #ADD NOISE
        return np.reshape(self._get_angle(),1,1)

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
